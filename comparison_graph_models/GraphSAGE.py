import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Load and preprocess data ===
df = pd.read_csv('/Users/beyzakaya/Desktop/temporalHiC/mapped/miRNA_expression_mean/standardized_time_columns_meaned_expression_values_get_closest.csv')
df = df.loc[:, ~df.columns.str.contains('Time_154.0')]

# Extract time columns and sorted unique time points
time_cols = [col for col in df.columns if 'Gene1_Time_' in col or 'Gene2_Time_' in col]
times = sorted(list(set([float(col.split('_')[-1]) for col in time_cols])))
print(f"Time points: {times}")

genes = sorted(set(df['Gene1']).union(set(df['Gene2'])))
gene2idx = {gene: idx for idx, gene in enumerate(genes)}
n_genes = len(genes)
print(f"Number of genes: {n_genes}")

# Build edges (undirected) from HiC interaction > median threshold
edge_list = []
for _, row in df.iterrows():
    g1, g2 = gene2idx[row['Gene1']], gene2idx[row['Gene2']]
    if row['HiC_Interaction'] > df['HiC_Interaction'].median():
        edge_list.append([g1, g2])
        edge_list.append([g2, g1])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Build gene expression matrix (genes x times)
expr_matrix = np.zeros((n_genes, len(times)))
for idx, t in enumerate(times):
    col_g1 = f'Gene1_Time_{t}'
    col_g2 = f'Gene2_Time_{t}'
    temp = np.zeros(n_genes)
    counts = np.zeros(n_genes)
    for _, row in df.iterrows():
        temp[gene2idx[row['Gene1']]] += row[col_g1]
        counts[gene2idx[row['Gene1']]] += 1
        temp[gene2idx[row['Gene2']]] += row[col_g2]
        counts[gene2idx[row['Gene2']]] += 1
    temp /= np.maximum(counts, 1)
    expr_matrix[:, idx] = temp

# Standardize per gene
scaler = StandardScaler()
expr_matrix = scaler.fit_transform(expr_matrix.T).T
print(f"Expression matrix shape: {expr_matrix.shape}")
print(f"Expression matrix sample: {expr_matrix[:, :5]}")  # Show first 5 time points

# Prepare sequences of length 'window_size' to predict next time point
window_size = 10
X, Y = [], []
for i in range(len(times) - window_size):
    X.append(expr_matrix[:, i:i+window_size])  # genes x window_size
    Y.append(expr_matrix[:, i+window_size])    # genes

X = np.stack(X)  # samples x genes x window_size
Y = np.stack(Y)  # samples x genes

print(f"Input shape: {X.shape}, Output shape: {Y.shape}")

# Train-test split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Train shape: {train_X.shape}, Test shape: {test_X.shape}")

# Convert to torch tensors and permute to [genes, samples, window_size]
train_X_t = torch.tensor(train_X, dtype=torch.float32).permute(1, 0, 2)  # genes x samples x window_size
test_X_t = torch.tensor(test_X, dtype=torch.float32).permute(1, 0, 2)
train_Y_t = torch.tensor(train_Y, dtype=torch.float32).permute(1, 0)    # genes x samples
test_Y_t = torch.tensor(test_Y, dtype=torch.float32).permute(1, 0)

edge_index = edge_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_X_t, train_Y_t = train_X_t.to(device), train_Y_t.to(device)
test_X_t, test_Y_t = test_X_t.to(device), test_Y_t.to(device)
edge_index = edge_index.to(device)

# === Define GraphSAGE model ===
class GraphSAGENet(torch.nn.Module):
    def __init__(self, window_size, hidden_channels=64):
        super(GraphSAGENet, self).__init__()
        self.window_size = window_size
        self.sage1 = SAGEConv(1, hidden_channels)
        self.sage2 = SAGEConv(hidden_channels, hidden_channels)
        self.linear = torch.nn.Linear(hidden_channels * window_size, 1)

    def forward(self, x, edge_index):
        xs = []
        for t in range(self.window_size):
            x_t = x[:, :, t]  # [num_nodes, 1], do NOT squeeze here
            h = F.relu(self.sage1(x_t, edge_index))
            h = F.relu(self.sage2(h, edge_index))
            xs.append(h)
        h_concat = torch.cat(xs, dim=1)
        out = self.linear(h_concat).squeeze()
        return out


model = GraphSAGENet(window_size).to(device)
optimizer = torch.optim.Adam(model.parameters(),  lr=0.0008, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

# === Training ===
epochs = 100
for epoch in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()
    loss = 0

    for i in range(train_X_t.shape[1]):  # iterate over samples
        x_t = train_X_t[:, i, :].unsqueeze(1)  # [num_nodes, 1, window_size]
        y_true = train_Y_t[:, i]                # [num_nodes]
        y_pred = model(x_t, edge_index)
        loss += criterion(y_pred, y_true)

    loss /= train_X_t.shape[1]
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Evaluation ===
model.eval()
preds = []
with torch.no_grad():
    for i in range(test_X_t.shape[1]):
        x_t = test_X_t[:, i, :].unsqueeze(1)  # [num_nodes, 1, window_size]
        y_pred = model(x_t, edge_index)
        preds.append(y_pred.cpu().numpy())

preds = np.stack(preds).T  # shape [num_nodes, samples]
true = test_Y_t.cpu().numpy()

# Flatten for correlation calculation
preds_flat = preds.flatten()
true_flat = true.flatten()

pearson_corr, _ = pearsonr(preds_flat, true_flat)
spearman_corr, _ = spearmanr(preds_flat, true_flat)

print(f"Pearson correlation: {pearson_corr:.4f}")
print(f"Spearman correlation: {spearman_corr:.4f}")

# === Plot some example genes ===
genes_to_plot = [0, 1, 2]
plt.figure(figsize=(12, 6))
for gene_idx in genes_to_plot:
    plt.plot(preds[gene_idx], label=f'Predicted Gene {gene_idx}')
    plt.plot(true[gene_idx], '--', label=f'True Gene {gene_idx}')
plt.xlabel('Test Sample')
plt.ylabel('Expression (scaled)')
plt.title('Predicted vs True Expression on Test Set (GraphSAGE)')
plt.legend()
plt.show()

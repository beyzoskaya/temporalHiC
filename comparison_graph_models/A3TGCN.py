import pandas as pd
import numpy as np
import torch
from torch_geometric_temporal.nn.recurrent import A3TGCN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm

# === Load and preprocess data ===

df = pd.read_csv('/Users/beyzakaya/Desktop/temporalHiC/mapped/miRNA_expression_mean/standardized_time_columns_meaned_expression_values_get_closest.csv')
df = df.loc[:, ~df.columns.str.contains('Time_154.0')]  # example to drop a time column

# Extract time columns and unique sorted time points
time_cols = [col for col in df.columns if 'Gene1_Time_' in col or 'Gene2_Time_' in col]
times = sorted(list(set([float(col.split('_')[-1]) for col in time_cols])))
print(f"Time points: {times}")

# Map genes to indices
genes = sorted(set(df['Gene1']).union(set(df['Gene2'])))
gene2idx = {gene: idx for idx, gene in enumerate(genes)}
n_genes = len(genes)
print(f"Number of genes: {n_genes}")

# Build undirected edge list from HiC interactions above median
edge_list = []
median_interaction = df['HiC_Interaction'].median()
for _, row in df.iterrows():
    if row['HiC_Interaction'] > median_interaction:
        g1 = gene2idx[row['Gene1']]
        g2 = gene2idx[row['Gene2']]
        edge_list.append([g1, g2])
        edge_list.append([g2, g1])

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Build expression matrix: genes x times
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

# Standardize per gene (across time)
scaler = StandardScaler()
expr_matrix = scaler.fit_transform(expr_matrix.T).T

print(f"Expression matrix shape: {expr_matrix.shape}")

# Prepare sequences of length window_size to predict next time point
window_size = 10
X, Y = [], []
for i in range(len(times) - window_size):
    X.append(expr_matrix[:, i:i+window_size])  # shape: genes x window_size
    Y.append(expr_matrix[:, i+window_size])    # shape: genes

X = np.stack(X)  # shape: samples x genes x window_size
Y = np.stack(Y)  # shape: samples x genes

print(f"Input shape: {X.shape}, Output shape: {Y.shape}")

# Train-test split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
print(f"Train shape: {train_X.shape}, Test shape: {test_X.shape}")

# Convert to torch tensors
train_X = torch.tensor(train_X, dtype=torch.float32)
train_Y = torch.tensor(train_Y, dtype=torch.float32)
test_X = torch.tensor(test_X, dtype=torch.float32)
test_Y = torch.tensor(test_Y, dtype=torch.float32)
edge_index = edge_index

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_X, train_Y = train_X.to(device), train_Y.to(device)
test_X, test_Y = test_X.to(device), test_Y.to(device)
edge_index = edge_index.to(device)

# === Define the A3TGCN model ===

class A3TGCNModel(torch.nn.Module):
    def __init__(self, out_channels=64, window_size=window_size):
        super(A3TGCNModel, self).__init__()
        # in_channels=1 because each node has 1 feature (expression) per timestep
        self.a3tgcn = A3TGCN(in_channels=1, out_channels=out_channels, periods=window_size)
        self.linear = torch.nn.Linear(out_channels, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, edge_index):
        # x shape: [num_nodes, in_channels=1, periods=window_size]
        out = self.a3tgcn(x, edge_index)  # shape: [num_nodes, out_channels]
        out = self.relu(out)
        out = self.linear(out).squeeze()  # shape: [num_nodes]
        return out

model = A3TGCNModel().to(device)
optimizer = torch.optim.Adam(model.parameters(),  lr=0.0008, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

# === Training ===

epochs = 100
for epoch in tqdm(range(epochs)):
    model.train()
    optimizer.zero_grad()
    loss = 0

    for i in range(train_X.shape[0]):
        x_t = train_X[i].unsqueeze(1)  # [num_nodes, 1, window_size]
        y_true = train_Y[i]            # [num_nodes]

        y_pred = model(x_t, edge_index)
        loss += criterion(y_pred, y_true)

    loss /= train_X.shape[0]
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Evaluation ===

model.eval()
preds = []

with torch.no_grad():
    for i in range(test_X.shape[0]):
        x_t = test_X[i].unsqueeze(1)  # [num_nodes, 1, window_size]
        y_pred = model(x_t, edge_index)
        preds.append(y_pred.cpu().numpy())

preds = np.stack(preds)
true = test_Y.cpu().numpy()

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
    plt.plot(preds[:, gene_idx], label=f'Predicted Gene {gene_idx}')
    plt.plot(true[:, gene_idx], '--', label=f'True Gene {gene_idx}')

plt.xlabel('Test Sample')
plt.ylabel('Expression (scaled)')
plt.title('Predicted vs True Expression on Test Set (A3TGCN)')
plt.legend()
plt.show()

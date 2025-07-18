import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data, TemporalData
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df = pd.read_csv('/Users/beyzakaya/Desktop/temporalHiC/mapped/miRNA_expression_mean/standardized_time_columns_meaned_expression_values_get_closest.csv')
df = df.loc[:, ~df.columns.str.contains('Time_154.0')]
time_cols = [col for col in df.columns if 'Gene1_Time_' in col or 'Gene2_Time_' in col]

times = sorted(list(set([float(col.split('_')[-1]) for col in time_cols])))
print(f"Time points: {times}")

genes = sorted(set(df['Gene1']).union(set(df['Gene2'])))
gene2idx = {gene: idx for idx, gene in enumerate(genes)}
n_genes = len(genes)
print(f"Number of genes: {n_genes}")


edge_list = []
for i, row in df.iterrows():
    g1, g2 = gene2idx[row['Gene1']], gene2idx[row['Gene2']]
    # optional: use HiC_Interaction to weight edges
    if row['HiC_Interaction'] > df['HiC_Interaction'].median():
        edge_list.append([g1, g2])
        edge_list.append([g2, g1])  # undirected

edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

# Extract time series matrix: genes x time
expr_matrix = np.zeros((n_genes, len(times)))
for idx, t in enumerate(times):
    col_g1 = f'Gene1_Time_{t}'
    col_g2 = f'Gene2_Time_{t}'
    # aggregate: average duplicate edges
    temp = np.zeros(n_genes)
    counts = np.zeros(n_genes)
    for _, row in df.iterrows():
        temp[gene2idx[row['Gene1']]] += row[col_g1]
        counts[gene2idx[row['Gene1']]] += 1
        temp[gene2idx[row['Gene2']]] += row[col_g2]
        counts[gene2idx[row['Gene2']]] += 1
    temp /= np.maximum(counts, 1)
    expr_matrix[:, idx] = temp

scaler = StandardScaler()
expr_matrix = scaler.fit_transform(expr_matrix.T).T  # genes x time
print(f"Expression matrix shape: {expr_matrix.shape}")
print(f"Expression matrix (first 5 genes, first 5 time points):\n{expr_matrix[:5, :5]}")

window_size = 10  
X, Y = [], []

for i in range(len(times) - window_size):
    X.append(expr_matrix[:, i:i+window_size])  # shape: genes x window_size
    Y.append(expr_matrix[:, i+window_size])    # shape: genes

X = np.stack(X)  # samples x genes x window_size
Y = np.stack(Y)  # samples x genes

print(f"Input shape: {X.shape}, Output shape: {Y.shape} ")

class TGCN(nn.Module):
    def __init__(self, n_genes, window_size, hidden_dim=256):
        super(TGCN, self).__init__()
        self.conv1 = GCNConv(window_size, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, 128)
        self.conv3 = GCNConv(128, 1)

    def forward(self, x, edge_index):
        # x: (genes x window_size)
        x = self.conv1(x, edge_index)  # (genes x hidden_dim)
        x = F.relu(x)
        x = self.conv2(x, edge_index)  # (genes x 1)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x.squeeze()  # (genes,)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TGCN(n_genes, window_size).to(device)
optimizer = torch.optim.Adam(model.parameters(),  lr=0.0008, weight_decay=1e-4)
criterion = nn.MSELoss()

train_X, test_X, train_Y, test_Y = train_test_split(
    X, Y, test_size=0.2, random_state=42  
)

train_X = torch.tensor(train_X, dtype=torch.float32).to(device)
train_Y = torch.tensor(train_Y, dtype=torch.float32).to(device)
test_X = torch.tensor(test_X, dtype=torch.float32).to(device)
test_Y = torch.tensor(test_Y, dtype=torch.float32).to(device)
edge_index = edge_index.to(device)

for epoch in tqdm(range(100)):
    model.train()
    optimizer.zero_grad()
    preds = []
    for i in range(train_X.shape[0]):
        out = model(train_X[i], edge_index)
        preds.append(out)
    preds = torch.stack(preds)
    loss = criterion(preds, train_Y)
    loss.backward()
    optimizer.step()
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')

model.eval()
with torch.no_grad():
    preds = []
    for i in range(test_X.shape[0]):
        out = model(test_X[i], edge_index)
        preds.append(out.cpu().numpy())
    preds = np.stack(preds)

preds_flat = preds.flatten()
print(f"Predictions shape: {preds.shape}")
true_flat = test_Y.cpu().numpy().flatten()
print(f"True values shape: {true_flat.shape}")

pearson_corr, _ = pearsonr(preds_flat, true_flat)
spearman_corr, _ = spearmanr(preds_flat, true_flat)

print(f'Pearson correlation: {pearson_corr:.4f}')
print(f'Spearman correlation: {spearman_corr:.4f}')


genes_to_plot = [0, 1, 2]  # first three genes as example

# preds and true_Y shape: (num_test_samples, n_genes)
test_Y_np = test_Y.cpu().numpy()

plt.figure(figsize=(12, 6))

for gene_idx in genes_to_plot:
    plt.plot(preds[:, gene_idx], label=f'Predicted Gene {gene_idx}')
    plt.plot(test_Y_np[:, gene_idx], '--', label=f'True Gene {gene_idx}')

plt.xlabel('Test Sample (Time Point)')
plt.ylabel('Expression (scaled)')
plt.title('Predicted vs True Expression Over Test Samples')
plt.legend()
plt.show()


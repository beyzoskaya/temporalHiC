import pandas as pd
import numpy as np
import torch
import networkx as nx
from gensim.models import Word2Vec
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch_geometric_temporal.nn.recurrent import TGCN
from node2vec import Node2Vec

# === Load and preprocess data ===
df = pd.read_csv('/Users/beyzakaya/Desktop/temporalHiC/mapped/miRNA_expression_mean/standardized_time_columns_meaned_expression_values_get_closest.csv')
df = df.loc[:, ~df.columns.str.contains('Time_154.0')]

time_cols = [col for col in df.columns if 'Gene1_Time_' in col or 'Gene2_Time_' in col]
times = sorted(list(set([float(col.split('_')[-1]) for col in time_cols])))
print(f"Time points: {times}")

genes = sorted(set(df['Gene1']).union(set(df['Gene2'])))
gene2idx = {gene: idx for idx, gene in enumerate(genes)}
idx2gene = {idx: gene for gene, idx in gene2idx.items()}
n_genes = len(genes)
print(f"Number of genes: {n_genes}")

# Build networkx graph
G = nx.Graph()
G.add_nodes_from(range(n_genes))
for _, row in df.iterrows():
    g1, g2 = gene2idx[row['Gene1']], gene2idx[row['Gene2']]
    if row['HiC_Interaction'] > df['HiC_Interaction'].median():
        G.add_edge(g1, g2)

# Build PyTorch edge_index
edge_list = list(G.edges())
edge_index = torch.tensor(edge_list + [(j, i) for i, j in edge_list], dtype=torch.long).t().contiguous()

# Build expression matrix
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

# Standardize
scaler = StandardScaler()
expr_matrix = scaler.fit_transform(expr_matrix.T).T

print(f"Expression matrix shape: {expr_matrix.shape}")

# Prepare sequences
window_size = 10
X, Y = [], []
for i in range(len(times) - window_size):
    X.append(expr_matrix[:, i:i+window_size])
    Y.append(expr_matrix[:, i+window_size])

X = np.stack(X)
Y = np.stack(Y)

print(f"Input shape: {X.shape}, Output shape: {Y.shape}")

# Train-test split
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)
train_X = torch.tensor(train_X, dtype=torch.float32)
train_Y = torch.tensor(train_Y, dtype=torch.float32)
test_X = torch.tensor(test_X, dtype=torch.float32)
test_Y = torch.tensor(test_Y, dtype=torch.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_X, train_Y, test_X, test_Y = train_X.to(device), train_Y.to(device), test_X.to(device), test_Y.to(device)
edge_index = edge_index.to(device)

walk_length = 20
num_walks = 10
dimensions = 32
window = 5
min_count = 1

node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, workers=2, seed=42)
model_n2v = node2vec.fit(window=window, min_count=min_count, batch_words=4)

# Build static node features: shape [num_nodes, embedding_dim]
static_node_features = np.zeros((n_genes, dimensions))
for node_idx in range(n_genes):
    static_node_features[node_idx] = model_n2v.wv[str(node_idx)]
static_node_features = torch.tensor(static_node_features, dtype=torch.float32).to(device)
print(f"Static node features shape: {static_node_features.shape}")

# === Define the model ===
class TemporalGCNModel(torch.nn.Module):
    def __init__(self, window_size, embedding_dim, hidden_size=128):
        super(TemporalGCNModel, self).__init__()
        total_features = window_size + embedding_dim
        self.tgcn1 = TGCN(total_features, hidden_size)
        self.tgcn2 = TGCN(hidden_size, 64)
        self.linear = torch.nn.Linear(64, 1)
        self.relu = torch.nn.ReLU()

    def forward(self, x, static_features, edge_index, h1=None, h2=None):
        combined = torch.cat([x, static_features], dim=1)
        h1 = self.tgcn1(combined, edge_index, h1)
        h1 = self.relu(h1)
        h2 = self.tgcn2(h1, edge_index, h2)
        h2 = self.relu(h2)
        out = self.linear(h2).squeeze()
        return out, (h1, h2)

model = TemporalGCNModel(window_size, dimensions).to(device)
optimizer = torch.optim.Adam(model.parameters(),  lr=0.0008, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

# === Training ===
for epoch in tqdm(range(100)):
    model.train()
    optimizer.zero_grad()
    loss = 0

    for i in range(train_X.shape[0]):
        x_t = train_X[i]  # [num_nodes, window_size]
        y_true = train_Y[i]
        h1, h2 = None, None
        y_pred, _ = model(x_t, static_node_features, edge_index, h1, h2)
        loss += criterion(y_pred, y_true)

    loss = loss / train_X.shape[0]
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Evaluation ===
model.eval()
preds = []
with torch.no_grad():
    for i in range(test_X.shape[0]):
        x_t = test_X[i]
        y_pred, _ = model(x_t, static_node_features, edge_index)
        preds.append(y_pred.cpu().numpy())

preds = np.stack(preds)
true = test_Y.cpu().numpy()

# Correlations
pearson_corr, _ = pearsonr(preds.flatten(), true.flatten())
spearman_corr, _ = spearmanr(preds.flatten(), true.flatten())

print(f"Pearson: {pearson_corr:.4f}")
print(f"Spearman: {spearman_corr:.4f}")

# Plot
plt.figure(figsize=(12,6))
for gene_idx in [0,1,2]:
    plt.plot(preds[:, gene_idx], label=f'Predicted Gene {gene_idx}')
    plt.plot(true[:, gene_idx], '--', label=f'True Gene {gene_idx}')
plt.title('Predicted vs True')
plt.legend()
plt.show()

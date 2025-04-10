import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import random

def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 

data = pd.read_csv('/Users/beyzakaya/Desktop/temporalHiC/mapped/mRNA/enhanced_interactions_synthetic_simple_mRNA.csv')

time_series_columns = [col for col in data.columns if 'Time' in col and 'Gene1' in col or 'Time' in col and 'Gene2' in col]
#print(f"Time Series columns: {time_series_columns}")

gene1_columns = [col for col in time_series_columns if 'Gene1' in col]
#print(f"Gene1 columns: {gene1_columns}")
gene2_columns = [col for col in time_series_columns if 'Gene2' in col]
#print(f"Gene2 columns: {gene2_columns}")

expression_data = data[gene1_columns].values
#print(f"Expression data: {expression_data}")

train_indices = [22, 9, 32, 15, 0, 3, 8, 18, 14, 13, 38, 2, 7, 4, 23, 37, 27, 29, 35, 17, 19, 25, 6, 21, 12, 10, 16, 39, 24, 33, 11, 34]
val_indices = [28, 20, 26, 31, 30, 36, 1, 5]

train_data = expression_data[train_indices]
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
std[std == 0] = 1  # prevent division by zero

normalized_expression_data = (expression_data - mean) / std

class GeneExpressionDataset(Dataset):
    def __init__(self, data, indices, seq_len, pred_len):
        self.data = data[indices]
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        label = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label.flatten(), dtype=torch.float32)

seq_len = 3  
pred_len = 1 
batch_size = 8

train_dataset = GeneExpressionDataset(normalized_expression_data, train_indices, seq_len, pred_len)
val_dataset = GeneExpressionDataset(normalized_expression_data, val_indices, seq_len, pred_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

class BasicMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=2):
        super().__init__()
        print(f"Input dimension: {input_dim}")
        self.attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.fc = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        attn_output, _ = self.attention(x, x, x)
        out = attn_output[:, -1, :]  # Take last timestep
        return self.fc(out)

class ProjectedMultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads=2):
        super().__init__()
        # Find nearest multiple of num_heads
        self.proj_dim = ((input_dim + num_heads - 1) // num_heads) * num_heads
        
        self.project_in = nn.Linear(input_dim, self.proj_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.proj_dim,
            num_heads=num_heads,
            batch_first=True
        )
        self.project_out = nn.Linear(self.proj_dim, input_dim)

    def forward(self, x):
        x_proj = self.project_in(x)
        attn_out, _ = self.attention(x_proj, x_proj, x_proj)
        return self.project_out(attn_out[:, -1, :])


input_dim = expression_data.shape[1]  # Number of genes/features
model = ProjectedMultiHeadAttention(input_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(50): 
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        #loss = criterion(outputs, labels[:, -1, :])  # Compare with last time step
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')



model.eval()
correlations = []
spearman_correlations = []
predictions = []
targets = []
with torch.no_grad():
    for batch in val_loader:
        inputs, labels = batch
        outputs = model(inputs)
        
        # Compare with last time step
        for i in range(outputs.shape[1]):
            gene_pred = outputs[:, i].cpu().numpy()
            gene_label = labels[:, i].cpu().numpy()
            
            # Calculate Pearson and Spearman correlations
            corr, _ = pearsonr(gene_pred, gene_label)
            correlations.append(corr)
            
            spearman_corr, _ = spearmanr(gene_pred, gene_label)
            spearman_correlations.append(spearman_corr)
        
        predictions.extend(outputs.cpu().numpy().flatten())
        targets.extend(labels.cpu().numpy().flatten())

def calculate_overall_metrics(predictions, targets):
    metrics = {}

    metrics['MSE'] = mean_squared_error(targets, predictions)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(targets, predictions)
    metrics['R2_Score'] = r2_score(targets, predictions)
    metrics['Pearson_Correlation'], _ = pearsonr(targets, predictions)
    
    return metrics

def plot_predictions(predictions, targets, gene_indices, num_genes_to_plot=5):
    plt.figure(figsize=(15, 10))
    
    for i, gene_idx in enumerate(gene_indices[:num_genes_to_plot]):
        plt.subplot(1, num_genes_to_plot, i + 1)
        plt.plot(targets[:, gene_idx], label='Original', color='blue')
        plt.plot(predictions[:, gene_idx], label='Predicted', color='orange')
        plt.title(f'Gene {gene_idx}')
        plt.xlabel('Time Steps')
        plt.ylabel('Expression Value')
        plt.legend()
    
    plt.tight_layout()
    plt.show()

overall_metrics = calculate_overall_metrics(predictions, targets)

print("Overall Metrics:")
for metric, value in overall_metrics.items():
    print(f"{metric}: {value}")

mean_pearson_corr = np.mean(correlations)
mean_spearman_corr = np.mean(spearman_correlations)

predictions_array = np.array(predictions).reshape(-1, len(expression_data[0]))
targets_array = np.array(targets).reshape(-1, len(expression_data[0]))

gene_indices_to_plot = list(range(predictions_array.shape[1]))

plot_predictions(predictions_array, targets_array, gene_indices_to_plot)

print(f"Mean Pearson Correlation: {mean_pearson_corr}")
print(f"Mean Spearman Correlation: {mean_spearman_corr}")
"""
Overall Metrics:
MSE: 0.399712473154068
RMSE: 0.6322281956672668
MAE: 0.5011242032051086
R2_Score: -0.5479233684742604
Pearson_Correlation: -0.4474722927215161
Mean Pearson Correlation: -0.48358597580851403
Mean Spearman Correlation: -0.19069767441860463
"""
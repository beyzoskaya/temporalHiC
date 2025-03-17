import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

data = pd.read_csv('mapped/miRNA_expression_mean/standardized_time_columns_meaned_expression_values_get_closest_without_biological_features.csv')
columns_to_drop = [col for col in data.columns if 'Time_154.0' in col]
data = data.drop(columns=columns_to_drop)

time_series_columns = [col for col in data.columns if 'Time' in col and 'Gene1' in col or 'Time' in col and 'Gene2' in col]
#print(f"Time Series columns: {time_series_columns}")

gene1_columns = [col for col in time_series_columns if 'Gene1' in col]
#print(f"Gene1 columns: {gene1_columns}")
gene2_columns = [col for col in time_series_columns if 'Gene2' in col]
#print(f"Gene2 columns: {gene2_columns}")

expression_data = data[gene1_columns].values
#print(f"Expression data: {expression_data}")

class GeneExpressionDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        seq = self.data[idx:idx + self.seq_len]
        #print(f"Sequence: {seq}")
        #print(f"Shape of sequence: {seq.shape}")
        label = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len]
        #print(f"Label shape: {label.shape}")
        #print(f"Label: {label}")
        
        return torch.tensor(seq, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

seq_len = 10  
pred_len = 1 
batch_size = 16

dataset = GeneExpressionDataset(expression_data, seq_len, pred_len)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

class BasicLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BasicLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

input_dim = expression_data.shape[1]  # Number of genes/features
print(f"Input dim: {input_dim}")
hidden_dim = 64
output_dim = input_dim

model = BasicLSTM(input_dim, hidden_dim, output_dim)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0009)

# Train the model
for epoch in range(60): 
    for batch in data_loader:
        inputs, labels = batch
        # No need to permute here since batch_first=True in LSTM
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels[:, -1, :])  # Compare with last time step
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')



model.eval()
correlations = []
spearman_correlations = []
predictions = []
targets = []
with torch.no_grad():
    for batch in data_loader:
        inputs, labels = batch
        outputs = model(inputs)
        
        # Compare with last time step
        for i in range(outputs.shape[1]):
            gene_pred = outputs[:, i].cpu().numpy()
            gene_label = labels[:, -1, i].cpu().numpy()
            
            # Calculate Pearson and Spearman correlations
            corr, _ = pearsonr(gene_pred, gene_label)
            correlations.append(corr)
            
            spearman_corr, _ = pearsonr(gene_pred, gene_label)  
            spearman_correlations.append(spearman_corr)
        
        predictions.extend(outputs.cpu().numpy().flatten())
        targets.extend(labels[:, -1, :].cpu().numpy().flatten())

def calculate_overall_metrics(predictions, targets):
    metrics = {}

    metrics['MSE'] = mean_squared_error(targets, predictions)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(targets, predictions)
    metrics['R2_Score'] = r2_score(targets, predictions)
    metrics['Pearson_Correlation'], _ = pearsonr(targets, predictions)
    
    return metrics

overall_metrics = calculate_overall_metrics(predictions, targets)

print("Overall Metrics:")
for metric, value in overall_metrics.items():
    print(f"{metric}: {value}")

mean_pearson_corr = np.mean(correlations)
mean_spearman_corr = np.mean(spearman_correlations)

print(f"Mean Pearson Correlation: {mean_pearson_corr}")
print(f"Mean Spearman Correlation: {mean_spearman_corr}")
"""
Overall Metrics:
MSE: 187481.328125
RMSE: 432.99114990234375
MAE: 102.66471862792969
R2_Score: 0.0028786799978195443
Pearson_Correlation: 0.1770166659513928
Mean Pearson Correlation: 0.3414051075257645
Mean Spearman Correlation: 0.3414051075257645
"""


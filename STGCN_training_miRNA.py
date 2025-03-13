import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import networkx as nx
import matplotlib.pyplot as plt
import os
import seaborn as sns
from node2vec import Node2Vec
from scipy.stats import pearsonr,spearmanr
#from STGCN.model.models import *
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
#sys.path.append('./STGCN')
#from STGCN.model.models import  STGCNChebGraphConvProjectedGeneConnectedAttentionLSTM,STGCNChebGraphConvProjectedGeneConnectedMultiHeadAttentionLSTMmirna,STGCNChebGraphConvProjectedGeneConnectedTransformerAttentionMirna,STGCNChebGraphConvProjectedGeneConnectedTransformerAttentionMirnaConnectionWeights
from model.models import  STGCNChebGraphConvProjectedGeneConnectedAttentionLSTM,STGCNChebGraphConvProjectedGeneConnectedMultiHeadAttentionLSTMmirna,STGCNChebGraphConvProjectedGeneConnectedTransformerAttentionMirna,STGCNChebGraphConvProjectedGeneConnectedTransformerAttentionMirnaConnectionWeights,BiLSTMExpressionPredictor,MultiHeadAttentionPredictor
import argparse
import random
from scipy.spatial.distance import cdist
from create_graph_and_embeddings_STGCN import *
from create_graph_and_embeddings_STGCN_mirna import *
from create_graph_and_embeddings_STGCN_mirna_wo_biological_features import *
from STGCN_losses import temporal_loss_for_projected_model, enhanced_temporal_loss, miRNA_enhanced_temporal_loss
from evaluation import *
from clustering_by_expr_levels import analyze_expression_levels_kmeans, analyze_expression_levels,analyze_expression_levels_research
from categorize_genes.ppi_calculation import get_mouse_ppi_data, compare_ppi_with_hic, get_mgi_info, analyze_and_plot_ppi, analyze_ppi_with_aliases


def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed) 
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42) 

def process_batch(seq, label):
    """Process batch data for training."""
    # Input: Use full embeddings
    x = torch.stack([g.x for g in seq])  # [seq_len, num_nodes, features]
    x = x.permute(2, 0, 1).unsqueeze(0)  # [1, features, seq_len, nodes]
    
    # Target: Use only expression values
    target = torch.stack([g.x[:, -1] for g in label])  # [1, nodes] (expression values)
    target = target.unsqueeze(1).unsqueeze(0)  # [1, 1, 1, nodes]
    
    return x, target

def compute_gene_correlations(dataset, model):
    sequences, labels = dataset.get_temporal_sequences()
    edge_index = sequences[0][0].edge_index
    all_targets = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for seq, label in zip(sequences, labels):
            x, target = process_batch(seq, label)
            output = model(x)
            all_targets.append(target.squeeze().cpu().numpy())  # [nodes]
            all_predictions.append(output[:, :, -1, :].squeeze().cpu().numpy())  # Use last time step pred length is 1

    targets = np.stack(all_targets, axis=0)  # [time_points, nodes]
    predictions = np.stack(all_predictions, axis=0)  # [time_points, nodes]

    print("Targets Shape gene correl:", targets.shape)
    print("Predictions Shape gene correl:", predictions.shape)

    gene_correlations = np.array([np.corrcoef(predictions[:, i], targets[:, i])[0, 1] for i in range(targets.shape[1])])
    return torch.tensor(gene_correlations, dtype=torch.float32)

def compute_gene_connections(dataset):
    connections = {}
    for idx, gene in enumerate(dataset.node_map.keys()):
        count1 = len(dataset.df[dataset.df['Gene1'] == gene])
        count2 = len(dataset.df[dataset.df['Gene2'] == gene])
        connections[idx] = float(count1 + count2)
    return connections

def train_stgcn(dataset,val_ratio=0.2):
    args = Args_miRNA()
    args.n_vertex = dataset.num_nodes
    n_vertex = dataset.num_nodes

    sequences, labels = dataset.get_temporal_sequences()
    print(f"\nCreated {len(sequences)} sequences")

    # GSO
    edge_index = sequences[0][0].edge_index
    edge_weight = sequences[0][0].edge_attr.squeeze() if sequences[0][0].edge_attr is not None else None

    adj = torch.zeros((args.n_vertex, args.n_vertex)) # symmetric matrix
    adj[edge_index[0], edge_index[1]] = 1 if edge_weight is None else edge_weight # diagonal vs nondiagonal elements for adj matrix
    D = torch.diag(torch.sum(adj, dim=1) ** (-0.5))
    args.gso = torch.eye(args.n_vertex) - D @ adj @ D
    
    train_sequences, train_labels, val_sequences, val_labels, train_idx, val_idx = dataset.split_sequences(sequences, labels)
    
    with open('plottings_STGCN_clustered/split_indices.txt', 'w') as f:
        f.write("Train Indices:\n")
        f.write(", ".join(map(str, train_idx)) + "\n")
        f.write("\nValidation Indices:\n")
        f.write(", ".join(map(str, val_idx)) + "\n")

    #model = STGCNChebGraphConvProjected(args, args.blocks, args.n_vertex)
    gene_connections = compute_gene_connections(dataset)
    
    model = STGCNChebGraphConvProjectedGeneConnectedMultiHeadAttentionLSTMmirna(args, args.blocks_temporal_node2vec_with_three_st_blocks_256dim, args.n_vertex, gene_connections)
    model = model.float() # convert model to float otherwise I am getting type error

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0009, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    criterion = nn.MSELoss()

    gene_correlations = compute_gene_correlations(dataset, model)
    print("Gene Correlations:", gene_correlations)
    print("Min Correlation:", gene_correlations.min().item())
    print("Max Correlation:", gene_correlations.max().item())
    print("Mean Correlation:", gene_correlations.mean().item())

    num_epochs = 60
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    save_dir = 'plottings_STGCN_clustered'
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_stats = []
        all_targets = []
        all_outputs = []

        for seq,label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            x,target = process_batch(seq, label)
            #x, _ = process_batch(seq, label)
            #print(f"Shape of Inpute inside training: {x.shape}") # --> [1, 32, 5, 52]
            #print(f"Shape of Target inside training: {target.shape}") # --> [1, 32, 1, 52]
            output = model(x)
            #print(f"Shape of output: {output.shape}") # --> [1, 32, 5, 52]
            #_, target = process_batch(seq, label)
            #target = target[:,:,-1:, :]
            #print(f"Shape of target: {target.shape}") # --> [32, 1, 52]

            # Don't take the last point for temporal loss !!!!!!
            #target = target[:, :, -1:, :]  # Keep only the last timestep
            #output = output[:, :, -1:, :]

            batch_stats.append({
                'target_range': [target.min().item(), target.max().item()],
                'output_range': [output.min().item(), output.max().item()],
                'target_mean': target.mean().item(),
                'output_mean': output.mean().item()
            })

            all_targets.append(target.detach().cpu().numpy())
            all_outputs.append(output.detach().cpu().numpy())
            #loss = enhanced_temporal_loss(
            # output[:, :, -1:, :],
            # target,
            # x
            #)

            loss = miRNA_enhanced_temporal_loss(
             output[:, :, -1:, :],
             target,
             x
            )

            #loss = criterion(output[:, :, -1:, :], target)
            if torch.isnan(loss):
                print("NaN loss detected!")
                print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        val_loss = 0
        val_loss_total = 0

        with torch.no_grad():
            for seq,label in zip(val_sequences, val_labels):
                #x, _ = process_batch(seq, label)
                x,target = process_batch(seq, label)
                
                output = model(x)
               
                #_, target = process_batch(seq, label)

                # Don't take the last point for temporal loss!!!
                #target = target[:, :, -1:, :]  
                #output = output[:, :, -1:, :]

                #print(f"Shape of output in validation: {output.shape}") # --> [1, 32, 5, 52]
                #print(f"Shape of target in validation: {target.shape}") # --> [32, 1, 52]
                #target = target[:,:,-1:, :]
                #val_loss = criterion(output[:, :, -1:, :], target)

                #val_loss = enhanced_temporal_loss(output[:, :, -1:, :], target, x)
                val_loss = miRNA_enhanced_temporal_loss(output[:, :, -1:, :], target, x)

                val_loss_total += val_loss.item()

                output_corr = calculate_correlation(output)
                #print(f"Shape of output corr: {output_corr.shape}") # [32, 32]
                target_corr = calculate_correlation(target)
                #print(f"Shape of target corr: {target_corr.shape}") # [32, 32]

        avg_train_loss = total_loss / len(train_sequences)
        avg_val_loss = val_loss_total / len(val_sequences)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}')
        print(f'Validation Loss: {avg_val_loss:.4f}')

        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
            }, f'{save_dir}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    checkpoint = torch.load(f'{save_dir}/best_model.pth', weights_only=True)
    #checkpoint = torch.load(f'{save_dir}/best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plot training progress
    plt.figure(figsize=(10, 5))
    
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    #plt.savefig(f'{save_dir}/training_progress.png')
    plt.close()
    
    return model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels

def train_stgcn_check_overfitting(dataset, val_ratio=0.2):
    args = Args_miRNA()
    args.n_vertex = dataset.num_nodes
    n_vertex = dataset.num_nodes

    sequences, labels = dataset.get_temporal_sequences()
    print(f"\nCreated {len(sequences)} sequences")

    # GSO
    edge_index = sequences[0][0].edge_index
    edge_weight = sequences[0][0].edge_attr.squeeze() if sequences[0][0].edge_attr is not None else None

    adj = torch.zeros((args.n_vertex, args.n_vertex)) # symmetric matrix
    adj[edge_index[0], edge_index[1]] = 1 if edge_weight is None else edge_weight # diagonal vs nondiagonal elements for adj matrix
    D = torch.diag(torch.sum(adj, dim=1) ** (-0.5))
    args.gso = torch.eye(args.n_vertex) - D @ adj @ D
    
    train_sequences, train_labels, val_sequences, val_labels, train_idx, val_idx = dataset.split_sequences(sequences, labels)
    
    with open('plottings_STGCN_clustered/split_indices.txt', 'w') as f:
        f.write("Train Indices:\n")
        f.write(", ".join(map(str, train_idx)) + "\n")
        f.write("\nValidation Indices:\n")
        f.write(", ".join(map(str, val_idx)) + "\n")

    gene_connections = compute_gene_connections(dataset)
    model = STGCNChebGraphConvProjectedGeneConnectedMultiHeadAttentionLSTMmirna(args, args.blocks_temporal_node2vec_option_two, args.n_vertex, gene_connections)
    model = model.float() # convert model to float otherwise I am getting type error

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0009, weight_decay=1e-4)

    gene_correlations = compute_gene_correlations(dataset, model)
    print("Initial Gene Correlations:", gene_correlations)
    print("Min Correlation:", gene_correlations.min().item())
    print("Max Correlation:", gene_correlations.max().item())
    print("Mean Correlation:", gene_correlations.mean().item())

    num_epochs = 30
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    save_dir = 'plottings_STGCN_clustered'
    os.makedirs(save_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    
    train_correlations = []
    val_correlations = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        batch_stats = []
        all_train_targets = []
        all_train_outputs = []

        for seq, label in zip(train_sequences, train_labels):
            optimizer.zero_grad()
            x, target = process_batch(seq, label)
            output = model(x)

            batch_stats.append({
                'target_range': [target.min().item(), target.max().item()],
                'output_range': [output.min().item(), output.max().item()],
                'target_mean': target.mean().item(),
                'output_mean': output.mean().item()
            })

            all_train_targets.append(target.detach().cpu().numpy())
            all_train_outputs.append(output[:, :, -1:, :].detach().cpu().numpy())
            
            loss = enhanced_temporal_loss(
             output[:, :, -1:, :],
             target,
             x
            )

            if torch.isnan(loss):
                print("NaN loss detected!")
                print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
                print(f"Target range: [{target.min().item():.4f}, {target.max().item():.4f}]")
                continue

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation loop
        model.eval()
        val_loss_total = 0
        all_val_targets = []
        all_val_outputs = []
        
        with torch.no_grad():
            for seq, label in zip(val_sequences, val_labels):
                x, target = process_batch(seq, label)
                output = model(x)
                
                # Save outputs and targets for correlation calculation
                all_val_targets.append(target.detach().cpu().numpy())
                all_val_outputs.append(output[:, :, -1:, :].detach().cpu().numpy())

                val_loss = enhanced_temporal_loss(output[:, :, -1:, :], target, x)
                val_loss_total += val_loss.item()

        avg_train_loss = total_loss / len(train_sequences)
        avg_val_loss = val_loss_total / len(val_sequences)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        train_corr = calculate_epoch_correlation(all_train_outputs, all_train_targets, dataset)
        val_corr = calculate_epoch_correlation(all_val_outputs, all_val_targets, dataset)
        
        train_correlations.append(train_corr)
        val_correlations.append(val_corr)
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')
        print(f'Training Correlation: {train_corr:.4f}, Validation Correlation: {val_corr:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'train_corr': train_corr,
                'val_corr': val_corr,
            }, f'{save_dir}/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    checkpoint = torch.load(f'{save_dir}/best_model.pth', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    plt.figure(figsize=(12, 5))
 
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_correlations, label='Train')
    plt.plot(val_correlations, label='Validation')
    plt.title('Training and Validation Correlation')
    plt.xlabel('Epoch')
    plt.ylabel('Correlation')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/training_metrics.png')
    plt.show()
    plt.close()
    
    is_overfitting = check_overfitting(train_correlations, val_correlations, train_losses, val_losses)
    if is_overfitting:
        print("Warning: Model shows signs of overfitting. Consider regularization or reducing model complexity.")
    
    return model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels

def calculate_epoch_correlation(all_outputs, all_targets, dataset):

    processed_outputs = []
    processed_targets = []
    
    for output, target in zip(all_outputs, all_targets):
        #print(f"Output shape: {output.shape}") --> [1, 1, 1, 162]
        #print(f"Target shape: {target.shape}") --> [1, 1, 1, 162]

        output_exp = output.squeeze()  
        target_exp = target.squeeze()  

        #print(f"Output squeezed shape: {output_exp.shape}") --> [162, ]
        #print(f"Target squeezed shape: {target_exp.shape}") --> [162, ]
        
        processed_outputs.append(output_exp)
        processed_targets.append(target_exp)
    
    #print(f"Processed outputs length: {len(processed_outputs)}, Processed targets length: {len(processed_targets)}")
    #print(f"Example processed output shape: {processed_outputs[0].shape if len(processed_outputs) > 0 else 'Empty'}")
    #print(f"Example processed target shape: {processed_targets[0].shape if len(processed_targets) > 0 else 'Empty'}")
    
    # Stack along the time dimension - this should give [time_points, nodes]
    stacked_outputs = np.vstack(processed_outputs)
    stacked_targets = np.vstack(processed_targets)

    #print(f"Output shape stacked: {stacked_outputs.shape}") --> [120, 162]
    #print(f"Target shape stacked: {stacked_targets.shape}") --> [120, 162]
    
    gene_correlations = []
    genes = list(dataset.node_map.keys())
    
    for gene_idx in range(stacked_outputs.shape[1]):
        pred_gene = stacked_outputs[:, gene_idx]  # All timepoints for this gene
        true_gene = stacked_targets[:, gene_idx]  # All timepoints for this gene

        #print(f"Pred gene shape: {pred_gene.shape}")
        #print(f"True gene shape: {true_gene.shape}")
     
        if np.std(pred_gene) > 0 and np.std(true_gene) > 0:
            corr, _ = pearsonr(pred_gene, true_gene)
            gene_correlations.append(corr)
        else:
            gene_correlations.append(0.0)  # No correlation if either has zero variance
    
    return np.mean(gene_correlations)

def check_overfitting(train_corrs, val_corrs, train_losses, val_losses):

    if len(train_corrs) > 10:
        recent_train_corr = np.mean(train_corrs[-5:])
        recent_val_corr = np.mean(val_corrs[-5:])
        
        corr_gap = recent_train_corr - recent_val_corr
      
        recent_train_loss = np.mean(train_losses[-5:])
        recent_val_loss = np.mean(val_losses[-5:])
        
        loss_ratio = recent_val_loss / recent_train_loss if recent_train_loss > 0 else 1.0

        if corr_gap > 0.2 or loss_ratio > 1.5:
            return True
    
    return False

def evaluate_model_performance(model, val_sequences, val_labels, dataset,save_dir='plottings_STGCN_clustered'):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x, target = process_batch(seq, label)
            output = model(x)
    
            # Extract only the expression predictions
            output = output[:, :, -1:, :].squeeze().cpu().numpy()  # [nodes] expression values
            target = target.squeeze().cpu().numpy()  # [nodes] expression values
            
            all_predictions.append(output)
            all_targets.append(target)

    predictions = np.array(all_predictions)  # [time_points, nodes]
    targets = np.array(all_targets)      # [time_points, nodes]
    
    overall_metrics = calculate_overall_metrics(predictions, targets)
    gene_metrics = calculate_gene_metrics(predictions, targets, dataset)
    temporal_metrics = calculate_temporal_metrics_detailly(predictions, targets, dataset)

    create_evaluation_plots(predictions, targets, dataset, save_dir)
    
    metrics = {
        'Overall': overall_metrics,
        'Gene': gene_metrics,
        'Temporal': temporal_metrics
    }
    
    return metrics

def calculate_overall_metrics(predictions, targets):
    """Calculate overall expression prediction metrics."""
    metrics = {}

    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    metrics['MSE'] = mean_squared_error(target_flat, pred_flat)
    metrics['RMSE'] = np.sqrt(metrics['MSE'])
    metrics['MAE'] = mean_absolute_error(target_flat, pred_flat)
    metrics['R2_Score'] = r2_score(target_flat, pred_flat)
    metrics['Pearson_Correlation'], _ = pearsonr(target_flat, pred_flat)
    
    return metrics

def calculate_gene_metrics(predictions, targets, dataset):
    """Calculate gene-specific metrics."""
    metrics = {}
    genes = list(dataset.node_map.keys())
    
    # Per-gene correlations
    gene_correlations = []
    gene_rmse = []
    gene_spearman_correlations = []

    for gene_idx, gene in enumerate(genes):
        pred_gene = predictions[:, gene_idx]  # All timepoints for this gene
        true_gene = targets[:, gene_idx]
        
        corr, _ = pearsonr(pred_gene, true_gene)
        spearman_corr, spearman_p = spearmanr(pred_gene, true_gene)
        rmse = np.sqrt(mean_squared_error(true_gene, pred_gene))
        
        gene_correlations.append((gene, corr))
        gene_spearman_correlations.append((gene, spearman_corr))
        gene_rmse.append(rmse)
    
    # Sort genes by correlation
    gene_correlations.sort(key=lambda x: x[1], reverse=True)
    gene_spearman_correlations.sort(key=lambda x: x[1], reverse=True)
    
    metrics['Mean_Correlation'] = np.mean([corr for _, corr in gene_correlations])
    metrics['Mean_Spearman_Correlation'] = np.mean([corr for _, corr in gene_spearman_correlations])
    metrics['Best_Genes_Pearson'] = [gene for gene, _ in gene_correlations[:5]]
    metrics['Best_Genes_Spearman'] = [gene for gene, _ in gene_spearman_correlations[:5]]
    metrics['Gene_RMSE'] = {gene: rmse for gene, rmse in zip(genes, gene_rmse)}
    
    return metrics

def calculate_temporal_metrics(predictions, targets, dataset):
    """Calculate temporal prediction metrics."""
    metrics = {}
    
    # Calculate changes between consecutive timepoints
    true_changes = np.diff(targets, axis=0)  # [time-1, nodes]
    pred_changes = np.diff(predictions, axis=0)
    
    # Direction accuracy (whether changes are in the same direction)
    direction_match = np.sign(true_changes) == np.sign(pred_changes)
    metrics['Direction_Accuracy'] = np.mean(direction_match)
    
    # Magnitude of changes
    metrics['Mean_True_Change'] = np.mean(np.abs(true_changes))
    metrics['Mean_Pred_Change'] = np.mean(np.abs(pred_changes))
    
    # Temporal correlation per gene
    genes = list(dataset.node_map.keys())
    temporal_corrs = []
    
    for gene_idx, gene in enumerate(genes):
        true_seq = targets[:, gene_idx]
        pred_seq = predictions[:, gene_idx]
        corr, _ = pearsonr(true_seq, pred_seq)
        temporal_corrs.append(corr)
    
    metrics['Mean_Temporal_Correlation'] = np.mean(temporal_corrs)
    
    return metrics

def calculate_temporal_metrics_detailly(predictions, targets, dataset):
    """Calculate temporal prediction metrics with more appropriate temporal measures."""
    metrics = {}
    

    """
    What if we shift one sequence by sequence length amount?
    """
    # 1. Time-lagged Cross Correlation
    def time_lagged_correlation(true_seq, pred_seq, max_lag=3):
        correlations = []
        for lag in range(max_lag + 1):
            if lag == 0:
                corr = np.corrcoef(true_seq, pred_seq)[0, 1]
            else:
                corr = np.corrcoef(true_seq[lag:], pred_seq[:-lag])[0, 1]
            correlations.append(corr)
        return np.max(correlations)  # Return max correlation across lags
    
    """
    measures the similarity between two temporal sequences by finding the optimal alignment between them

    True sequence:  [1, 2, 3, 2, 1]
    Pred sequence: [1, 1, 2, 3, 1]

    DTW process:
    1. Creates a matrix of distances
    2. For each point, calculates:
    - Direct cost (difference between values)
    - Adds minimum cost from previous steps
    3. Finds optimal path through matrix that minimizes total distance

    Visual example:
    True:  1 -> 2 -> 3 -> 2 -> 1
            \   |  /  |    /
    Pred:   1 -> 1 -> 2 -> 3 -> 1
    """
    # 2. Dynamic Time Warping Distance
    def dtw_distance(true_seq, pred_seq):
        n, m = len(true_seq), len(pred_seq)
        dtw_matrix = np.zeros((n+1, m+1))
        dtw_matrix[1:, 0] = np.inf
        dtw_matrix[0, 1:] = np.inf
        
        for i in range(1, n+1):
            for j in range(1, m+1):
                cost = abs(true_seq[i-1] - pred_seq[j-1])
                dtw_matrix[i, j] = cost + min(dtw_matrix[i-1, j],    # insertion
                                            dtw_matrix[i, j-1],    # deletion
                                            dtw_matrix[i-1, j-1])  # match
        return dtw_matrix[n, m]
    
    # Calculate temporal metrics for each gene
    genes = list(dataset.node_map.keys())
    temporal_metrics = []
    dtw_distances = []
    direction_accuracies = []
    
    for gene_idx, gene in enumerate(genes):
        true_seq = targets[:, gene_idx]
        pred_seq = predictions[:, gene_idx]
        
        # Time-lagged correlation
        temp_corr = time_lagged_correlation(true_seq, pred_seq)
        temporal_metrics.append(temp_corr)
        
        # DTW distance
        dtw_dist = dtw_distance(true_seq, pred_seq)
        dtw_distances.append(dtw_dist)
        
        # Direction of changes
        true_changes = np.diff(true_seq)
        pred_changes = np.diff(pred_seq)
        dir_acc = np.mean(np.sign(true_changes) == np.sign(pred_changes))
        direction_accuracies.append(dir_acc)
    
    metrics['Mean_Temporal_Correlation'] = np.mean(temporal_metrics)
    metrics['Mean_DTW_Distance'] = np.mean(dtw_distances)
    metrics['Mean_Direction_Accuracy'] = np.mean(direction_accuracies)
    
    # Calculate rate of change metrics
    true_changes = np.diff(targets, axis=0)
    pred_changes = np.diff(predictions, axis=0)
    
    metrics['Mean_True_Change'] = np.mean(np.abs(true_changes))
    metrics['Mean_Pred_Change'] = np.mean(np.abs(pred_changes))
    metrics['Change_Magnitude_Ratio'] = metrics['Mean_Pred_Change'] / metrics['Mean_True_Change']
    
    return metrics

def create_gene_temporal_plots(predictions, targets, dataset, save_dir):
    """Create temporal pattern plots for all genes across multiple pages."""
    genes = list(dataset.node_map.keys())
    genes_per_page = 15  # Show 15 genes per page (5x3 grid)
    num_genes = len(genes)
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page
    
    for page in range(num_pages):
        plt.figure(figsize=(20, 15))
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        current_genes = genes[start_idx:end_idx]
        
        for i, gene in enumerate(current_genes):
            plt.subplot(5, 3, i+1)
            gene_idx = dataset.node_map[gene]
            
            # Plot actual and predicted values
            plt.plot(targets[:, gene_idx], 'b-', label='Actual', marker='o')
            plt.plot(predictions[:, gene_idx], 'r--', label='Predicted', marker='s')
            
            # Calculate metrics for this gene
            corr, _ = pearsonr(targets[:, gene_idx], predictions[:, gene_idx])
            rmse = np.sqrt(mean_squared_error(targets[:, gene_idx], predictions[:, gene_idx]))
            
            # Calculate changes
            actual_changes = np.diff(targets[:, gene_idx])
            pred_changes = np.diff(predictions[:, gene_idx])
            direction_acc = np.mean(np.sign(actual_changes) == np.sign(pred_changes))
            
            plt.title(f'Gene: {gene}\nCorr: {corr:.3f}, RMSE: {rmse:.3f}\nDir Acc: {direction_acc:.3f}')
            plt.xlabel('Time Step')
            plt.ylabel('Expression')
            if i == 0:  # Only show legend for first plot
                plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/temporal_patterns_page_{page+1}.png')
        plt.close()

def create_evaluation_plots(predictions, targets, dataset, save_dir):
    """Create comprehensive evaluation plots."""
    # 1. Overall prediction scatter
    plt.figure(figsize=(10, 8))
    plt.scatter(targets.flatten(), predictions.flatten(), alpha=0.1)
    plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
    plt.xlabel('True Expression')
    plt.ylabel('Predicted Expression')
    plt.title('Expression Prediction Performance')
    plt.savefig(f'{save_dir}/overall_scatter.png')
    plt.close()
    
    # 2. Change distribution plot
    true_changes = np.diff(targets, axis=0).flatten()
    pred_changes = np.diff(predictions, axis=0).flatten()
    
    plt.figure(figsize=(12, 6))
    plt.hist(true_changes, bins=50, alpha=0.5, label='Actual Changes')
    plt.hist(pred_changes, bins=50, alpha=0.5, label='Predicted Changes')
    plt.xlabel('Expression Change')
    plt.ylabel('Frequency')
    plt.title('Distribution of Expression Changes')
    plt.legend()
    plt.savefig(f'{save_dir}/change_distribution.png')
    plt.close()
    
    # 3. Gene temporal patterns for all genes
    create_gene_temporal_plots(predictions, targets, dataset, save_dir)

def plot_gene_predictions_train_val(model, train_sequences, train_labels, val_sequences, val_labels, dataset,save_dir='plottings_STGCN_clustered', genes_per_page=12):

    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    all_sequences = train_sequences + val_sequences
    all_labels = train_labels + val_labels
    
    num_genes = dataset.num_nodes
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(all_sequences, all_labels):
            x, target = process_batch(seq, label)
            output = model(x)
            
            output = output[:, :, -1:, :].squeeze().cpu().numpy() 
            target = target.squeeze().cpu().numpy()  
            
            all_predictions.append(output)
            all_targets.append(target)
    
    predictions = np.array(all_predictions)  # [time_points, nodes]
    targets = np.array(all_targets)          # [time_points, nodes]
    
    gene_names = list(dataset.node_map.keys())
    
    num_pages = (num_genes + genes_per_page - 1) // genes_per_page

    for page in range(num_pages):
        plt.figure(figsize=(20, 15))  
        
        start_idx = page * genes_per_page
        end_idx = min((page + 1) * genes_per_page, num_genes)
        page_genes = gene_names[start_idx:end_idx]
        
        for i, gene_name in enumerate(page_genes):
            gene_idx = start_idx + i
            
            rows = (genes_per_page + 1) // 2  
            plt.subplot(rows, 2, i + 1) 
        
            train_time_points = range(len(train_labels))
            plt.plot(train_time_points, targets[:len(train_labels), gene_idx], label='Train Actual', color='blue', marker='o')
            
            plt.plot(train_time_points, predictions[:len(train_labels), gene_idx], label='Train Predicted', color='red', linestyle='--', marker='x')
     
            val_time_points = range(len(train_labels), len(train_labels) + len(val_labels))
            plt.plot(val_time_points, targets[len(train_labels):, gene_idx], label='Val Actual', color='green', marker='o')
            
            plt.plot(val_time_points, predictions[len(train_labels):, gene_idx], label='Val Predicted', color='orange', linestyle='--', marker='x')
            
            plt.title(f'Gene: {gene_name}')
            plt.xlabel('Time Points')
            plt.ylabel('Expression Value')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'{save_dir}/gene_predictions_page_{page + 1}.png')
        plt.close()

def get_predictions_and_targets(model, val_sequences, val_labels):
    """Extract predictions and targets from validation data."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for seq, label in zip(val_sequences, val_labels):
            x, target = process_batch(seq, label)
            output = model(x)
            
            # Take last time step for predictions
            output = output[:, :, -1:, :]
            
            # Convert to numpy and reshape
            pred = output.squeeze().cpu().numpy()  # Should be [52] or [1, 52]
            true = target.squeeze().cpu().numpy()  # Should be [52] or [1, 52]
            
            if len(pred.shape) == 1:
                pred = pred.reshape(1, -1)
            if len(true.shape) == 1:
                true = true.reshape(1, -1)
            
            all_predictions.append(pred)
            all_targets.append(true)

    predictions = np.vstack(all_predictions)  # Should be [8, 52]
    targets = np.vstack(all_targets)        # Should be [8, 52]
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    return predictions, targets

def analyze_gene_characteristics(dataset, predictions, targets):
    """Analyze relationship between gene properties and prediction performance"""
    genes = list(dataset.node_map.keys())
    
    # Calculate gene correlations
    gene_correlations = {}
    for gene in genes:
        gene_idx = dataset.node_map[gene]
        pred_gene = predictions[:, gene_idx]  # [time_points]
        true_gene = targets[:, gene_idx]
        corr, _ = pearsonr(pred_gene, true_gene)
        gene_correlations[gene] = corr
    
    # Collect gene properties
    gene_stats = {gene: {
        'degree': len(dataset.base_graph[gene]),
        'expression_range': None,
        'expression_std': None,
        'correlation': gene_correlations[gene]
    } for gene in genes}
    
    # Calculate expression statistics
    for gene in genes:
        all_expressions = []
        for t in dataset.time_points:
            gene1_expr = dataset.df[dataset.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = dataset.df[dataset.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_values = np.concatenate([gene1_expr, gene2_expr])
            all_expressions.extend(expr_values)
        
        all_expressions = np.array(all_expressions)
        gene_stats[gene].update({
            'expression_range': np.ptp(all_expressions),
            'expression_std': np.std(all_expressions)
        })
    
    # Create analysis plots
    plt.figure(figsize=(15, 10))
    
    # 1. Correlation vs Degree
    plt.subplot(2, 2, 1)
    degrees = [gene_stats[gene]['degree'] for gene in genes]
    correlations = [gene_stats[gene]['correlation'] for gene in genes]
    plt.scatter(degrees, correlations)
    plt.xlabel('Number of Interactions')
    plt.ylabel('Prediction Correlation')
    plt.title('Gene Connectivity vs Prediction Performance')
    
    # 2. Correlation vs Expression Range
    plt.subplot(2, 2, 2)
    ranges = [gene_stats[gene]['expression_range'] for gene in genes]
    plt.scatter(ranges, correlations)
    plt.xlabel('Expression Range')
    plt.ylabel('Prediction Correlation')
    plt.title('Expression Variability vs Prediction Performance')
    
    plt.subplot(2, 2, 3)
    plt.hist(correlations, bins=20)
    plt.xlabel('Correlation')
    plt.ylabel('Count')
    plt.title('Distribution of Gene Correlations')
    
    plt.tight_layout()
    plt.savefig('plottings_STGCN_clustered/gene_analysis.png')
    plt.close()
    
    print("\nGene Analysis Summary:")
    print("\nTop 5 Most Connected Genes:")
    sorted_by_degree = sorted(gene_stats.items(), key=lambda x: x[1]['degree'], reverse=True)[:5]
    for gene, stats in sorted_by_degree:
        print(f"{gene}: {stats['degree']} connections, correlation: {stats['correlation']:.4f}")
    
    print("\nTop 5 Most Variable Genes:")
    sorted_by_range = sorted(gene_stats.items(), key=lambda x: x[1]['expression_range'], reverse=True)[:5]
    for gene, stats in sorted_by_range:
        print(f"{gene}: range {stats['expression_range']:.4f}, correlation: {stats['correlation']:.4f}")
    
    print("\nTop 5 Best Predicted Genes:")
    sorted_by_corr = sorted(gene_stats.items(), key=lambda x: x[1]['correlation'], reverse=True)[:5]
    for gene, stats in sorted_by_corr:
        print(f"{gene}: correlation {stats['correlation']:.4f}, connections: {stats['degree']}")
    
    print(f"\nCorrelation values for all genes:")
    sorted_by_corr_all_genes = sorted(gene_stats.items(), key=lambda x: x[1]['correlation'], reverse=True)
    for gene, stats in sorted_by_corr_all_genes:
        print(f"{gene}: correlation {stats['correlation']:.4f}, connections: {stats['degree']}")
    
    return gene_stats

def analyze_temporal_patterns(dataset, predictions, targets):
    time_points = dataset.time_points
    genes = list(dataset.node_map.keys())

    temporal_stats = {
        'prediction_lag': [],  # Time shift between predicted and actual peaks
        'pattern_complexity': [],  # Number of direction changes
        'prediction_accuracy': []  # Accuracy by time point
    }

    time_point_accuracy = []
    for t in range(len(predictions)):
        corr = pearsonr(predictions[t].flatten(), targets[t].flatten())[0]
        time_point_accuracy.append(corr)
    
    plt.figure(figsize=(15, 5))
    plt.plot(time_point_accuracy)
    plt.xlabel('Time Point')
    plt.ylabel('Prediction Accuracy')
    plt.title('Prediction Accuracy Over Time')
    plt.savefig(f'plottings_STGCN_clustered/pred_accuracy.png')

    print("\nTemporal Analysis:")
    print(f"Best predicted time point: {np.argmax(time_point_accuracy)}")
    print(f"Worst predicted time point: {np.argmin(time_point_accuracy)}")
    print(f"Mean accuracy: {np.mean(time_point_accuracy):.4f}")
    print(f"Std of accuracy: {np.std(time_point_accuracy):.4f}")
    
    return temporal_stats


"""
Kt and Ks control how model processes information across both time and space
If I want use more time point and more order for graph, I need to use higher value of n_his to not get negative value of Ko

n_his determines how far back in time model looks when making predictions
n_his directly determines how many consecutive time points of data model receives as input --> in miRNA expression prediction, this means how many previous time points of expression data the model can see.
"""
class Args_miRNA:
    def __init__(self):
        #self.Kt = 3 # temporal kernel size
        #self.Ks = 3 # spatial kernel size
        #self.n_his = 6 # number of historical time steps

        #self.Kt = 4
        #self.Ks = 4
        #self.n_his = 8 

        self.Kt=3
        self.Ks=3
        self.n_his=13
        #self.n_his=17 # I am changing number of historical time steps because number of ST blocks are increased, Ko become < 0!
        self.n_pred = 1
       
        self.blocks = [
             [64, 64, 64],    # Input block
             [64, 48, 48],    # Single ST block (since temporal dim reduces quickly)
             [48, 32, 1]      # Output block
        ]
        
        self.blocks_two_st = [
            [64, 64, 64],    
            [64, 32, 32], 
            [32, 48, 48],    
            [48, 32, 1]      
        ]

        self.blocks_temporal_node2vec = [
            [128, 128, 128],    
            [128, 64, 64], 
            [64, 48, 48],    
            [48, 32, 1]      
        ]

        self.blocks_temporal_node2vec_option_two = [
            [128, 128, 128],
            [128, 64, 64],
            [64, 96, 96],
            [96, 64, 1]
        ]

        self.blocks_temporal_node2vec_with_three_st_blocks = [
            [128, 128, 128],    # Initial block
            [128, 96, 96],      # First ST block output
            [96, 64, 64],       # Second ST block output
            [64, 96, 96],       # Third ST block output
            [96, 64, 1]         # Output block
        ]

        self.blocks_temporal_node2vec_with_three_st_blocks_256dim = [
            [256, 256, 256],    # Initial block (doubled from 128)
            [256, 192, 192],    # First ST block output (doubled from 96)
            [192, 128, 128],    # Second ST block output (doubled from 64)
            [128, 192, 192],    # Third ST block output (doubled from 96)
            [192, 128, 1]       # Output block (doubled intermediate dimension)
        ]

        self.blocks_temporal_node2vec_with_four_st_blocks = [
            [128, 128, 128],    # Initial block
            [128, 112, 112],    # First ST block output
            [112, 96, 96],      # Second ST block output
            [96, 64, 64],       # Third ST block output
            [64, 96, 96],       # Fourth ST block output
            [96, 64, 1]         # Output block
        ]

        self.act_func = 'gelu'
        self.graph_conv_type = 'cheb_graph_conv'
        self.enable_bias = True
        self.droprate = 0.2

if __name__ == "__main__":
    dataset = TemporalGraphDatasetMirna(
        csv_file = 'mapped/miRNA_expression_mean/standardized_time_columns_meaned_expression_values_get_closest.csv',
        embedding_dim=256,
        #seq_len=6,
        seq_len=13,
        pred_len=1
    )

    #dataset = TemporalGraphDatasetMirnaNoExtraBiological(
    #    csv_file = 'mapped/miRNA_expression_mean/standardized_time_columns_meaned_expression_values_get_closest_without_biological_features.csv',
    #    embedding_dim=256,
    #    #seq_len=6,
    #    seq_len=13,
    #    pred_len=1
    #)
   
    model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels = train_stgcn(dataset, val_ratio=0.2)
    #model, val_sequences, val_labels, train_losses, val_losses, train_sequences, train_labels = train_stgcn_check_overfitting(dataset, val_ratio=0.2)
    metrics = evaluate_model_performance(model, val_sequences, val_labels, dataset)
    plot_gene_predictions_train_val(model, train_sequences, train_labels, val_sequences, val_labels, dataset)

    print("\nModel Performance Summary:")
    print("\nOverall Metrics:")
    for metric, value in metrics['Overall'].items():
        print(f"{metric}: {value:.4f}")

    print("\nGene Performance:")
    print(f"Mean Gene Correlation: {metrics['Gene']['Mean_Correlation']:.4f}")
    print(f"Mean Spearman Correlation: {metrics['Gene']['Mean_Spearman_Correlation']:.4f}")
    print(f"Best Performing Genes Pearson: {', '.join(metrics['Gene']['Best_Genes_Pearson'])}")
    print(f"Best Performing Genes Spearman: {', '.join(metrics['Gene']['Best_Genes_Spearman'])}")

    print("\nTemporal Performance:")
    print(f"Time-lagged Correlation: {metrics['Temporal']['Mean_Temporal_Correlation']:.4f}")
    print(f"DTW Distance: {metrics['Temporal']['Mean_DTW_Distance']:.4f}")
    print(f"Direction Accuracy: {metrics['Temporal']['Mean_Direction_Accuracy']:.4f}")
    print(f"Change Magnitude Ratio: {metrics['Temporal']['Change_Magnitude_Ratio']:.4f}")

    predictions, targets = get_predictions_and_targets(model, val_sequences, val_labels)
    gene_stats = analyze_gene_characteristics(dataset, predictions, targets)
    temporal_stats = analyze_temporal_patterns(dataset, predictions, targets)

    gene_metrics = create_gene_analysis_plots(
    model, 
    train_sequences, 
    train_labels, 
    val_sequences, 
    val_labels, 
    dataset
)

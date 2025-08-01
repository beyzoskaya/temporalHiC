import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def temporal_pattern_loss(output, target, input_sequence):

    mse_loss = F.mse_loss(output, target)
    
    input_trend = input_sequence[:, :, 1:, :] - input_sequence[:, :, :-1, :]
    last_trend = input_sequence[:, :, -1, :] - input_sequence[:, :, -2, :]
    
    pred_trend = output - input_sequence[:, :, -1:, :]
    
    trend_loss = F.mse_loss(pred_trend, last_trend)
    
    return mse_loss + 0.1 * trend_loss

def change_magnitude_loss(output, target, input_sequence, alpha=1.0, beta=0.5):

    pred_loss = F.mse_loss(output, target)
    
    actual_change = target - input_sequence[:, :, -1:, :]
    pred_change = output - input_sequence[:, :, -1:, :]
    
    magnitude_loss = F.mse_loss(torch.abs(pred_change), torch.abs(actual_change))
    
    underpredict_penalty = torch.mean(
        torch.relu(torch.abs(actual_change) - torch.abs(pred_change))
    )
    
    return pred_loss + alpha * magnitude_loss + beta * underpredict_penalty

def temporal_loss_for_projected_model(output, target, input_sequence, alpha=0, gamma=0.4):
   
    mse_loss = F.mse_loss(output, target)
    
    # Get expression values from input sequence
    input_expressions = input_sequence[:, -1, :, :]  # [1, 3, 52]
    last_input = input_expressions[:, -1:, :]  # [1, 1, 52]
    
    # Reshape output and target
    output_reshaped = output.squeeze(1)  # [1, 1, 52]
    target_reshaped = target.squeeze(1)  # [1, 1, 52]
    
    # Direction loss
    true_change = target_reshaped - last_input
    pred_change = output_reshaped - last_input
    direction_match = torch.sign(true_change) * torch.sign(pred_change)
    direction_loss = -torch.mean(direction_match)  # Keep negative
    
    # Temporal correlation
    def trend_correlation_loss(pred, target, sequence_expr):
        pred_trend = torch.cat([sequence_expr, pred], dim=1)
        target_trend = torch.cat([sequence_expr, target], dim=1)
        
        pred_norm = (pred_trend - pred_trend.mean(dim=1, keepdim=True)) / (pred_trend.std(dim=1, keepdim=True) + 1e-8)
        target_norm = (target_trend - target_trend.mean(dim=1, keepdim=True)) / (target_trend.std(dim=1, keepdim=True) + 1e-8)
        
        return -torch.mean(torch.sum(pred_norm * target_norm, dim=1))
    
    
    temp_loss = trend_correlation_loss(output_reshaped, target_reshaped, input_expressions)
    
    # Combine losses
    total_loss = 0.6 * mse_loss + alpha * direction_loss + gamma * temp_loss
   
    # Print components for monitoring
    print(f"\nLoss Components:")
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"Direction Loss: {direction_loss.item():.4f}")
    print(f"Temporal Loss: {temp_loss.item():.4f}")
    
    return total_loss

def enhanced_temporal_loss(output, target, input_sequence, alpha=0.3, beta=0.3, gamma=0.4):
    mse_loss = F.mse_loss(output, target)
    l1_loss = F.l1_loss(output, target)
    
    # expression values from input sequence
    input_expressions = input_sequence[:, -1, :, :]  # [1, 3, 52]
    last_input = input_expressions[:, -1:, :]  # [1, 1, 52]
    
    output_reshaped = output.squeeze(1)  # [1, 1, 52]
    target_reshaped = target.squeeze(1)  # [1, 1, 52]

    true_change = target_reshaped - last_input
    pred_change = output_reshaped - last_input

    true_norm = F.normalize(true_change, p=2, dim=-1)
    pred_norm = F.normalize(pred_change, p=2, dim=-1)
    direction_cosine = torch.sum(true_norm * pred_norm, dim=-1)
    direction_loss = 1 - torch.mean(direction_cosine)
    direction_loss = direction_loss * 0.01

    def enhanced_trend_correlation(pred, target, sequence_expr):
        pred_trend = torch.cat([sequence_expr, pred], dim=1)
        target_trend = torch.cat([sequence_expr, target], dim=1)

        def correlation_loss(x, y):
            x_centered = x - x.mean(dim=1, keepdim=True)
            y_centered = y - y.mean(dim=1, keepdim=True)
            x_norm = torch.sqrt(torch.sum(x_centered**2, dim=1) + 1e-8)
            y_norm = torch.sqrt(torch.sum(y_centered**2, dim=1) + 1e-8)
            correlation = torch.sum(x_centered * y_centered, dim=1) / (x_norm * y_norm)
            return 1 - correlation.mean() # 1-correlation --> minimization in the loss func
        
        corr_loss = correlation_loss(pred_trend, target_trend)
        smoothness_loss = torch.mean(torch.abs(torch.diff(pred_trend, dim=1)))
        return corr_loss + 0.1 * smoothness_loss

    
    temporal_loss = enhanced_trend_correlation(output_reshaped, target_reshaped, input_expressions)
    temporal_loss = temporal_loss * 0.1
    
    last_sequence_val = input_expressions[:, -1, :]
    consistency_loss = torch.mean(torch.abs(output_reshaped - last_sequence_val))

    #direction_weight = 0.1 if direction_loss.item() > 0.1 else 0.05
    #temporal_weight = 0.3 if temporal_loss.item() > 0.2 else 0.1
    #consistency_weight = 0.3 if consistency_loss.item() > 0.3 else 0.2
    
    total_loss = (
        0.3 * mse_loss +
        0.2 * direction_loss + 
        0.2 * temporal_loss +
        0.3 * consistency_loss
    )

    print(f"\nLoss Components:")
    print(f"MSE Loss: {mse_loss.item():.4f}")
    #print(f"L1 loss: {l1_loss.item():.4f}")
    print(f"Direction Loss: {direction_loss.item():.4f}")
    print(f"Temporal Loss: {temporal_loss.item():.4f}")
    print(f"Consistency Loss: {consistency_loss.item():.4f}")
    
    return total_loss

# Try to train without scale parts for directional and temporal losses in the exact same settings
def miRNA_enhanced_temporal_loss(output, target, input_sequence, alpha=0.3, beta=0.2, gamma=0.3, delta=0.2):
    mse_loss = F.mse_loss(output, target)
    l1_loss = F.l1_loss(output, target)
 
    #input_expressions = input_sequence[:, -1, :, :]  # Feature dimension, time steps, genes
    input_expressions = input_sequence[:, :, :, -1] # for ASTGCN
    last_input = input_expressions[:, -1:, :]  # Last time point
   
    output_reshaped = output.squeeze(1)
    target_reshaped = target.squeeze(1)

    # Direction loss - captures if predictions move in the correct direction
    true_change = target_reshaped - last_input
    pred_change = output_reshaped - last_input

    # cosine similarity to measure directional agreement
    true_norm = F.normalize(true_change, p=2, dim=-1)
    pred_norm = F.normalize(pred_change, p=2, dim=-1)
    direction_cosine = torch.sum(true_norm * pred_norm, dim=-1)
    direction_loss = 1 - torch.mean(direction_cosine)
    
    # Scale direction loss to be comparable with other components
    scaled_direction_loss = direction_loss * 0.01
    
    def enhanced_trend_correlation(pred, target, sequence_expr):
        #pred_trend = torch.cat([sequence_expr, pred], dim=1)
        #target_trend = torch.cat([sequence_expr, target], dim=1)
        pred_trend = torch.cat([sequence_expr, pred], dim=2)  # for ASTGCN
        target_trend = torch.cat([sequence_expr, target], dim=2) # for ASTGCN

        def correlation_loss(x, y):
            x_centered = x - x.mean(dim=1, keepdim=True)
            y_centered = y - y.mean(dim=1, keepdim=True)
            x_norm = torch.sqrt(torch.sum(x_centered**2, dim=1) + 1e-8)
            y_norm = torch.sqrt(torch.sum(y_centered**2, dim=1) + 1e-8)
            correlation = torch.sum(x_centered * y_centered, dim=1) / (x_norm * y_norm + 1e-8)
            return 1 - correlation.mean()  # 1-correlation for minimization
        
        corr_loss = correlation_loss(pred_trend, target_trend)
        smoothness_loss = torch.mean(torch.abs(torch.diff(pred_trend, dim=1)))
       
        return corr_loss + 0.15 * smoothness_loss
 
    temporal_loss = enhanced_trend_correlation(output_reshaped, target_reshaped, input_expressions)
    scaled_temporal_loss = temporal_loss * 0.1
    
    # Consistency loss - encourages predictions to be realistic extensions of history
    last_sequence_val = input_expressions[:, -1, :]
    consistency_loss = torch.mean(torch.abs(output_reshaped - last_sequence_val))
    
    def simplified_correlation_structure(pred, target):
        #print(f"Pred dimension: {pred.dim()}")
        #print(f"Target dimension: {target.dim()}")
        if pred.dim() > 2:
            pred = pred.reshape(-1, pred.shape[-1])
        if target.dim() > 2:
            target = target.reshape(-1, target.shape[-1])
            
        # correlation between genes for pred and target
        pred_centered = pred - pred.mean(dim=0, keepdim=True)
        target_centered = target - target.mean(dim=0, keepdim=True)

        pred_var = torch.sum(pred_centered**2, dim=0, keepdim=True) / pred.shape[0] + 1e-8
        target_var = torch.sum(target_centered**2, dim=0, keepdim=True) / target.shape[0] + 1e-8
        
        pred_std = torch.sqrt(pred_var)
        target_std = torch.sqrt(target_var)
        #pred_std = torch.std(pred, dim=0, keepdim=True) + 1e-8
        #target_std = torch.std(target, dim=0, keepdim=True) + 1e-8
        
        pred_normalized = pred_centered / pred_std
        target_normalized = target_centered / target_std
        
        # gene-gene correlation matrices
        pred_corr = torch.mm(pred_normalized.t(), pred_normalized) / (pred.shape[0])
        target_corr = torch.mm(target_normalized.t(), target_normalized) / (target.shape[0])
        
        #return F.mse_loss(pred_corr, target_corr)
        return F.l1_loss(pred_corr, target_corr)
    
    total_loss = (
        alpha * l1_loss +
        beta * scaled_direction_loss + 
        gamma * scaled_temporal_loss +
        delta * consistency_loss
    )

    print(f"\nLoss Components:")
    #print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"L1 loss: {l1_loss.item():.4f}")
    print(f"Direction Loss: {direction_loss.item():.4f}")
    print(f"Temporal Loss: {temporal_loss.item():.4f}")
    print(f"Consistency Loss: {consistency_loss.item():.4f}")
   
    return total_loss

def gene_specific_loss(output, target, input_sequence, gene_correlations=None, alpha=0.2, beta=0.2, gamma=0.3):

    output = output[:, :, -1, :]  # [batch_size, 1, nodes]
    target = target[:, :, -1, :]  # [batch_size, 1, nodes]s
    mse_loss = F.mse_loss(output, target)
    #l1_loss = F.l1_loss(output, target)
    
    if gene_correlations is not None:
        gene_mse_loss = F.mse_loss(output, target, reduction='none').mean(dim=(0, 1))  # [nodes]
        #gene_mse_loss = F.l1_loss(output, target, reduction='none').mean(dim=(0, 1))

        #print("Gene-wise MSE Loss:", gene_mse_loss)
        #print("Min Gene-wise MSE:", gene_mse_loss.min().item())
        #print("Max Gene-wise MSE:", gene_mse_loss.max().item())
        #print("Mean Gene-wise MSE:", gene_mse_loss.mean().item())
        
        gene_weights = 1.0 / (gene_correlations**2 + 1e-8)
        gene_weights = gene_weights / gene_weights.sum()

        #print("Gene Weights:", gene_weights)
        #print("Min Weight:", gene_weights.min().item())
        #print("Max Weight:", gene_weights.max().item())
        #print("Mean Weight:", gene_weights.mean().item())

        gene_specific_loss = (gene_mse_loss * gene_weights).mean()
        gene_specific_loss = gene_specific_loss * 10
        #print("Gene-Specific Loss:", gene_specific_loss.item())
    else:
        #print(f"I'm inside gene correl is None")
        gene_specific_loss = 0.0

    input_expressions = input_sequence[:, -1, :, :]  # [1, 3, 52]
    last_input = input_expressions[:, -1:, :]  # [1, 1, 52]
    
    output_reshaped = output.squeeze(1)  # [1, 1, 52]
    target_reshaped = target.squeeze(1)  # [1, 1, 52]
    
    true_change = target_reshaped - last_input
    pred_change = output_reshaped - last_input

    true_norm = F.normalize(true_change, p=2, dim=-1)
    pred_norm = F.normalize(pred_change, p=2, dim=-1)
    direction_cosine = torch.sum(true_norm * pred_norm, dim=-1)
    direction_loss = 1 - torch.mean(direction_cosine)
    direction_loss = direction_loss * 0.01

    def enhanced_trend_correlation(pred, target, sequence_expr):
        pred = pred.unsqueeze(1)  # [1, 1, 52]
        target = target.unsqueeze(1)
        #print(f"Shape of pred inside enhanced trend correl: {pred.shape}")
        #print(f"Shape of target inside enhanced trend correl: {pred.shape}")
        pred_trend = torch.cat([sequence_expr, pred], dim=1)
        target_trend = torch.cat([sequence_expr, target], dim=1)

        def correlation_loss(x, y):
            x_centered = x - x.mean(dim=1, keepdim=True)
            y_centered = y - y.mean(dim=1, keepdim=True)
            x_norm = torch.sqrt(torch.sum(x_centered**2, dim=1) + 1e-8)
            y_norm = torch.sqrt(torch.sum(y_centered**2, dim=1) + 1e-8)
            correlation = torch.sum(x_centered * y_centered, dim=1) / (x_norm * y_norm)
            return 1 - correlation.mean()  # 1-correlation --> minimization in the loss func
        
        corr_loss = correlation_loss(pred_trend, target_trend)
        smoothness_loss = torch.mean(torch.abs(torch.diff(pred_trend, dim=1)))
        return corr_loss + 0.1 * smoothness_loss

    temporal_loss = enhanced_trend_correlation(output_reshaped, target_reshaped, input_expressions)
    temporal_loss = temporal_loss * 0.1
    
    last_sequence_val = input_expressions[:, -1, :]
    consistency_loss = torch.mean(torch.abs(output_reshaped - last_sequence_val))

    total_loss = (
        0.3 * mse_loss +
        0.3 * gene_specific_loss + 
        0.2 * temporal_loss +
        0.3 * consistency_loss
    )

    print(f"\nLoss Components:")
    print(f"MSE Loss: {mse_loss.item():.4f}")
    print(f"Gene-Specific Loss: {gene_specific_loss.item():.4f}")
    print(f"Direction Loss: {direction_loss.item():.4f}")
    print(f"Temporal Loss: {temporal_loss.item():.4f}")
    print(f"Consistency Loss: {consistency_loss.item():.4f}")
    
    return total_loss



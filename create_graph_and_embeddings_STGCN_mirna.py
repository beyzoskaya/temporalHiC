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
from scipy.stats import pearsonr
from model.models import *
import sys
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import argparse
from scipy.spatial.distance import cdist
from clustering_by_expr_levels import analyze_expression_levels_research, analyze_expression_levels_kmeans,analyze_expression_levels_gmm
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import QuantileTransformer
from networkx.algorithms.components import is_connected
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from torch.utils.data import DataLoader, TensorDataset
import random

# Temporal Node2Vec doesn't change the dimensional aspect of the model but change the creation of the embeddings in terms of capturing relations
# I changed the walk_length=25 num_walks=75 (this is the best version one lately)
class TemporalNode2Vec:
    def __init__(self, dimensions=256, walk_length=25, num_walks=75, p=1.0, q=1.0, workers=1, seed=42, temporal_weight=0.5): # temporal_weight 0.5 gave the best correlation value (from 0.6 it gets more overfit!!!)
        self.dimensions = dimensions
        print(f"Embedding dimension in TemporalNode2Vec: {self.dimensions}")
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.workers = workers
        self.seed = seed
        self.temporal_weight = temporal_weight
        
    def fit_single_graph(self, graph, window=5, min_count=1, batch_words=4):
        node2vec = Node2Vec(
            graph,
            dimensions=self.dimensions,
            walk_length=self.walk_length,
            num_walks=self.num_walks,
            p=self.p,
            q=self.q,
            workers=self.workers,
            seed=self.seed
        )
        
        model = node2vec.fit(window=window, min_count=min_count, batch_words=batch_words)
        return model
    
    def temporal_fit(self, temporal_graphs, time_points, node_map, window=5, min_count=1, batch_words=4):

        initial_embeddings = {}
        models = {}
        
        for t in time_points:
            print(f"\nInitial embedding for time point {t}")
            graph = temporal_graphs[t]
            model = self.fit_single_graph(graph, window, min_count, batch_words)
            models[t] = model
            initial_embeddings[t] = {node: model.wv[node] for node in graph.nodes()}
        
        # create temporal graph with weighted edges between time points
        temporal_graph = nx.Graph()
        
        # Add all nodes and edges from individual time graphs
        for t, graph in temporal_graphs.items():
            # Add nodes with time attribute
            for node in graph.nodes():
                temporal_graph.add_node(f"{node}_t{t}", original_node=node, time=t)
            
            # edges within the same time point (spatial edges)
            for u, v, data in graph.edges(data=True):
                temporal_graph.add_edge(
                    f"{u}_t{t}", 
                    f"{v}_t{t}", 
                    weight=data.get('weight', 1.0),
                    edge_type='spatial'
                )
        
        # temporal edges between consecutive time points
        for t_idx in range(len(time_points) - 1):
            t_curr = time_points[t_idx]
            t_next = time_points[t_idx + 1]
            
            # Find nodes present in both time points
            curr_nodes = set(temporal_graphs[t_curr].nodes())
            next_nodes = set(temporal_graphs[t_next].nodes())
            common_nodes = curr_nodes.intersection(next_nodes)
            
            # temporal edges with similarity-based weights
            for node in common_nodes:
                # Weight based on cosine similarity between embeddings at consecutive time points
                if node in initial_embeddings[t_curr] and node in initial_embeddings[t_next]:
                    embed_curr = initial_embeddings[t_curr][node]
                    embed_next = initial_embeddings[t_next][node]
                    
                    # cosine similarity
                    sim = np.dot(embed_curr, embed_next) / (
                        np.linalg.norm(embed_curr) * np.linalg.norm(embed_next) + 1e-8
                    )
                    
                    # temporal weight factor (higher values prioritize temporal consistency)
                    edge_weight = sim * self.temporal_weight
                    
                    # temporal edge
                    temporal_graph.add_edge(
                        f"{node}_t{t_curr}", 
                        f"{node}_t{t_next}", 
                        weight=edge_weight,
                        edge_type='temporal'
                    )
        
        # Node2Vec on the temporal graph
        print("\nFitting Node2Vec on temporal graph...")
        temporal_model = self.fit_single_graph(temporal_graph, window, min_count, batch_words)
        
        # temporal embeddings for each time point
        temporal_embeddings = {}
        temporal_embeddings_normalized = {}
        for t in time_points:
            embeddings = []
            normalized_embeddings = []

            for node in node_map.keys():
                temporal_node_name = f"{node}_t{t}"
                if temporal_node_name in temporal_model.wv:
                    embedding = torch.tensor(temporal_model.wv[temporal_node_name], dtype=torch.float32)
                else:
                    if node in initial_embeddings[t]:
                        embedding = torch.tensor(initial_embeddings[t][node], dtype=torch.float32)
                        print(f"Embeddings not found for {node} at different time {t}")
                    else:
                        # If node not found at all, use zeros
                        embedding = torch.zeros(self.dimensions, dtype=torch.float32)
                        print(f"Embeddings not found for {node} at time {t}")
                embeddings.append(embedding)

                embedding_norm = torch.norm(embedding, p=2) + 1e-8  
                normalized_embedding = embedding / embedding_norm
                normalized_embeddings.append(normalized_embedding)
            
            temporal_embeddings[t] = torch.stack(embeddings)
            temporal_embeddings_normalized[t] = torch.stack(normalized_embeddings)
            
            print(f"\nEmbedding statistics for time {t}:")
            print(f"Min: {temporal_embeddings[t].min().item():.4f}, Max: {temporal_embeddings[t].max().item():.4f}")
            print(f"Mean: {temporal_embeddings[t].mean().item():.4f}, Std: {temporal_embeddings[t].std().item():.4f}")

            print(f"\nNormalized Embedding statistics for time {t}:")
            print(f"Min: {temporal_embeddings_normalized[t].min().item():.4f}, Max: {temporal_embeddings_normalized[t].max().item():.4f}")
            print(f"Mean: {temporal_embeddings_normalized[t].mean().item():.4f}, Std: {temporal_embeddings_normalized[t].std().item():.4f}")
        
        return temporal_embeddings
    
class TemporalGraphDatasetMirna:
    def __init__(self, csv_file, embedding_dim=256, seq_len=10, pred_len=1, graph_params=None, node2vec_params=None): # I change the seq_len to more lower value
        #self.graph_params = graph_params or {}
        #self.node2vec_params = node2vec_params or {}
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.embedding_dim = embedding_dim
        
        self.df = pd.read_csv(csv_file)

        """
        I added this to not change proper parts for cleaning gene name because in miRNA data, all genes are in cleaned format
        """
        self.df['Gene1_clean'] = self.df['Gene1']
        self.df['Gene2_clean'] = self.df['Gene2']
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed', case=False)]

        unique_genes = pd.concat([self.df['Gene1_clean'], self.df['Gene2_clean']]).unique()
        self.node_map = {gene: idx for idx, gene in enumerate(unique_genes)}
        print(f"Unique genes: {self.node_map}")
        self.num_nodes = len(self.node_map)
        print(f"Number of nodes: {self.num_nodes}")
        
        # Get time points
        self.time_cols = [col for col in self.df.columns if 'Time_' in col]
        # This is for mRNA data because column names for time points are different
        self.time_points = sorted(list(set([float(col.split('_')[-1])  
                                          for col in self.time_cols if 'Gene1' in col])))
        self.time_points = [float(tp) for tp in self.time_points] # added for solving type error of time_points
      
        print(f"Found {len(self.time_points)} time points")
        print("Extracted time points:", self.time_points)

        self.time_points = [tp for tp in self.time_points if tp != 154.0]
        self.df = self.df.loc[:, ~self.df.columns.str.contains('Time_154.0', case=False)]
        print(f"After dropping time point 154.0, remaining time points: {self.time_points}")
        
        # Create base graph and features
        self.base_graph = self.create_base_graph()
        print("Base graph created")
        #self.node_features = self.create_temporal_node_features_several_graphs_created_clustering() # try with several graphs for time series consistency
        self.node_features, self.temporal_edge_indices, self.temporal_edge_attrs = \
        self.create_temporal_node_features_with_temporal_node2vec(debug_mode=True)
        print("Temporal node features created")
        
        # Get edge information
        self.edge_index, self.edge_attr = self.get_edge_index_and_attr()
        print(f"Graph structure created with {len(self.edge_attr)} edges")
    
    def create_base_graph(self):
        """Create a single base graph using structural features"""
        G = nx.Graph()
        G.add_nodes_from(self.node_map.keys())
        
        for _, row in self.df.iterrows():
            hic_weight = row['HiC_Interaction']
            compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
            tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
            tad_sim = 1 / (1 + tad_dist)
            ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
            
            # Use only structural features for base graph
            weight = (hic_weight * 0.4 + 
                     compartment_sim * 0.2 + 
                     tad_sim * 0.2 + 
                     ins_sim * 0.2)
            
            G.add_edge(row['Gene1_clean'], row['Gene2_clean'], weight=weight)
        
        return G
    
    def create_temporal_node_features_several_graphs_created_mirna(self, debug_mode=True):

        temporal_features = {}
        temporal_edge_indices = {}
        temporal_edge_attrs = {}

        clusters, _ = analyze_expression_levels_gmm(self)
        gene_clusters = {}
        for cluster_name, genes in clusters.items():
            for gene in genes:
                gene_clusters[gene] = cluster_name
        
        print("\nNormalizing expression values across all time points...")
        all_expressions = []
        log_expressions = []

        for t in self.time_points:
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                log_expr = np.log1p(expr_value + 1e-7)
                
                all_expressions.append(expr_value)
                log_expressions.append(log_expr)
                
        global_min = min(all_expressions)
        global_max = max(all_expressions)
        print(f"Global expression range: [{global_min:.4f}, {global_max:.4f}]") #Global expression range: [1.0000, 19592.0000] 

        for t in self.time_points:
            print(f"\nProcessing time point {t}")
            #print(f"All time points: {self.time_points}")
            
            expression_values = {}
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                if np.isnan(gene1_expr).any():
                    print(f"NaN detected in Gene1 expression for {gene} at time {t}")
                if np.isnan(gene2_expr).any():
                    print(f"NaN detected in Gene2 expression for {gene} at time {t}")

                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)

                expression_values[gene] = (expr_value - global_min) / (global_max - global_min + 1e-8)

            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())

            edge_index = []
            edge_weights = []
            for _, row in self.df.iterrows():
                gene1 = row['Gene1_clean']
                gene2 = row['Gene2_clean']
                
                expr_sim = 1 / (1 + abs(expression_values[gene1] - expression_values[gene2]))

                if np.isnan(expr_sim):
                    print(f"NaN detected in expression similarity between {gene1} and {gene2} at time {t}")
                
                hic_weight = row['HiC_Interaction'] 
                
                if pd.isna(row['HiC_Interaction']):
                    print(f"HiC weight is NaN")
            
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0 

                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))

                # Added for debugging of miRNA data
                if np.isnan(hic_weight):
                    print(f"NaN detected in HiC weight for {gene1}-{gene2}")
                if np.isnan(compartment_sim):
                    print(f"NaN detected in compartment similarity for {gene1}-{gene2}")
                if np.isnan(tad_sim):
                    print(f"NaN detected in TAD similarity for {gene1}-{gene2}")
                if np.isnan(ins_sim):
                    print(f"NaN detected in insulation similarity for {gene1}-{gene2}")
                
                if gene_clusters[gene1] == gene_clusters[gene2]:
                    cluster_sim = 1.2
                    #print(f"Same cluster: {gene1} ({gene_clusters[gene1]}) - {gene2} ({gene_clusters[gene2]}), Similarity: {cluster_sim}")
                else:
                    cluster_sim = 1.0
                    #print(f"Different clusters: {gene1} ({gene_clusters[gene1]}) - {gene2} ({gene_clusters[gene2]}), Similarity: {cluster_sim}")

                weight = (hic_weight * 0.3 +
                        compartment_sim * 0.1 +
                        tad_sim * 0.1 +
                        ins_sim * 0.1 +
                        expr_sim * 0.4)
                
                G.add_edge(gene1, gene2, weight=weight)
                i, j = self.node_map[gene1], self.node_map[gene2]
                edge_index.extend([[i, j], [j, i]])
                edge_weights.extend([weight, weight])
            
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim,
                walk_length=25,
                num_walks=75,
                p=1.0,
                q=1.0,
                workers=1,
                seed=42
            )
            
            model = node2vec.fit(
                window=5,
                min_count=1,
                batch_words=4
            )

            features = []
            for gene in self.node_map.keys():
                node_embedding = torch.tensor(model.wv[gene], dtype=torch.float32)
                #softplus for node embedding's negative values
                #node_embedding = torch.log(1 + torch.exp(node_embedding))
                min_val = node_embedding.min()
                print(f"Node embedding original value: {node_embedding}")

                print(f"\n{gene} embedding analysis:")
                print(f"Original last dimension value: {node_embedding[-1]:.4f}")

                #if min_val < 0:
                #    node_embedding = node_embedding - min_val  
                #    node_embedding = node_embedding / (node_embedding.max() + 1e-8)
                #print(f"Node embedding value shifting through zero and normalized: {node_embedding}")

                orig_mean = node_embedding.mean().item()
                orig_std = node_embedding.std().item()
                
                # Normalize embedding
                #node_embedding = (node_embedding - node_embedding.min()) / (node_embedding.max() - node_embedding.min() + 1e-8) #FIXME Normalization of embeddings not affect performance in a good way
                #print(f"Normalized last dimension value: {node_embedding[-1]:.4f}")

                #print(f"Expression value to be inserted: {expression_values[gene]:.4f}")

                orig_last_dim = node_embedding[-1].item()
    
                #print(f"Node embeddings last dim before adding expression value: {node_embedding[-1]}")
                #node_embedding[-1] = expression_values[gene]
                #print(f"Node embeddings last dim after adding expression value: {node_embedding[-1]}")
                #print(f"Expression value for {gene}: {node_embedding[-1]}")
                
                print(f"Statistics before override:")
                print(f"  Mean: {orig_mean:.4f}")
                print(f"  Std: {orig_std:.4f}")
                print(f"  Last dim: {orig_last_dim:.4f}")
                print(f"Statistics after override:")
                print(f"  Mean: {node_embedding.mean().item():.4f}")
                print(f"  Std: {node_embedding.std().item():.4f}")
                print(f"  Last dim: {node_embedding[-1].item():.4f}")
                print('*************************************************************')
               
                features.append(node_embedding)
            
            temporal_features[t] = torch.stack(features)
            temporal_edge_indices[t] = torch.tensor(edge_index).t().contiguous()
            temporal_edge_attrs[t] = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)

            print(f"\nFeature Statistics for time {t}:")
            print(f"Expression range: [{min(expression_values.values()):.4f}, {max(expression_values.values()):.4f}]")
            print(f"Embedding range: [{temporal_features[t].min():.4f}, {temporal_features[t].max():.4f}]")
            
        return temporal_features, temporal_edge_indices, temporal_edge_attrs
    
    def create_temporal_node_features_with_temporal_node2vec(self, debug_mode=True):
        #clusters, _ = analyze_expression_levels_gmm(self)
        clusters, _ = analyze_expression_levels_kmeans(self)
        gene_clusters = {}
        for cluster_name, genes in clusters.items():
            for gene in genes:
                gene_clusters[gene] = cluster_name
        
        print("\nNormalizing expression values across all time points...")
        all_expressions = []
        for t in self.time_points:
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                all_expressions.append(expr_value)
                
        global_min = min(all_expressions)
        global_max = max(all_expressions)
        print(f"Global expression range: [{global_min:.4f}, {global_max:.4f}]")
        
        temporal_graphs = {}
        temporal_edge_indices = {}
        temporal_edge_attrs = {}
        
        for t in self.time_points:
            print(f"\nProcessing time point {t}")
            
            expression_values = {}
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                
                expression_values[gene] = (expr_value - global_min) / (global_max - global_min + 1e-8)
            
            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())
            
            edge_index = []
            edge_weights = []
            
            for _, row in self.df.iterrows():
                gene1 = row['Gene1_clean']
                gene2 = row['Gene2_clean']
                
                expr_sim = 1 / (1 + abs(expression_values[gene1] - expression_values[gene2]))
                hic_weight = row['HiC_Interaction'] if not pd.isna(row['HiC_Interaction']) else 0
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
                
                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
                
                cluster_sim = 1.2 if gene_clusters.get(gene1) == gene_clusters.get(gene2) else 1.0
                
                weight = (hic_weight * 0.3 +
                        compartment_sim * 0.1 +
                        tad_sim * 0.1 +
                        ins_sim * 0.1 +
                        expr_sim * 0.4) 
                
                G.add_edge(gene1, gene2, weight=weight, time=t)
                
                i, j = self.node_map[gene1], self.node_map[gene2]
                edge_index.extend([[i, j], [j, i]])
                edge_weights.extend([weight, weight])
    
            temporal_graphs[t] = G
            temporal_edge_indices[t] = torch.tensor(edge_index).t().contiguous()
            temporal_edge_attrs[t] = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
        
        temporal_node2vec = TemporalNode2Vec(
            dimensions=self.embedding_dim,
            walk_length=25,
            num_walks=75,
            p=1.0,
            q=1.0,
            workers=1,
            seed=42,
            temporal_weight=0.5  
        )
        
        temporal_features = temporal_node2vec.temporal_fit(
            temporal_graphs=temporal_graphs,
            time_points=self.time_points,
            node_map=self.node_map,
            window=5,
            min_count=1,
            batch_words=4
        )
        
        for t in self.time_points:
            for i, gene in enumerate(self.node_map.keys()):
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
        
        for time_point, graph in temporal_graphs.items():
            print(f"Plotting graph for time point {time_point}")
            visualize_enhanced_gene_graph(base_graph=graph, gene_names=self.node_map, time_point=time_point)
        
        return temporal_features, temporal_edge_indices, temporal_edge_attrs
    
    def create_temporal_node_features_with_temporal_node2vec_original_node2vec(self, debug_mode=True):
        clusters, _ = analyze_expression_levels_gmm(self)
        gene_clusters = {}
        for cluster_name, genes in clusters.items():
            for gene in genes:
                gene_clusters[gene] = cluster_name
        
        print("\nNormalizing expression values across all time points...")
        all_expressions = []
        for t in self.time_points:
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                all_expressions.append(expr_value)
                
        global_min = min(all_expressions)
        global_max = max(all_expressions)
        print(f"Global expression range: [{global_min:.4f}, {global_max:.4f}]")
        
        # original (non-temporal) embeddings for each gene
        original_embeddings = {}
        
        temporal_graphs = {}
        temporal_edge_indices = {}
        temporal_edge_attrs = {}
        
        for t in self.time_points:
            print(f"\nProcessing time point {t}")
            
            expression_values = {}
            for gene in self.node_map.keys():
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                
                expression_values[gene] = (expr_value - global_min) / (global_max - global_min + 1e-8)
            
            G = nx.Graph()
            G.add_nodes_from(self.node_map.keys())
            
            edge_index = []
            edge_weights = []
            
            for _, row in self.df.iterrows():
                gene1 = row['Gene1_clean']
                gene2 = row['Gene2_clean']
                
                expr_sim = 1 / (1 + abs(expression_values[gene1] - expression_values[gene2]))
                hic_weight = row['HiC_Interaction'] if not pd.isna(row['HiC_Interaction']) else 0
                compartment_sim = 1 if row['Gene1_Compartment'] == row['Gene2_Compartment'] else 0
                
                tad_dist = abs(row['Gene1_TAD_Boundary_Distance'] - row['Gene2_TAD_Boundary_Distance'])
                tad_sim = 1 / (1 + tad_dist)
                ins_sim = 1 / (1 + abs(row['Gene1_Insulation_Score'] - row['Gene2_Insulation_Score']))
                
                cluster_sim = 1.2 if gene_clusters.get(gene1) == gene_clusters.get(gene2) else 1.0
                
                weight = (hic_weight * 0.3 +
                        compartment_sim * 0.1 +
                        tad_sim * 0.1 +
                        ins_sim * 0.1 +
                        expr_sim * 0.4) 
                
                G.add_edge(gene1, gene2, weight=weight, time=t)
                
                i, j = self.node_map[gene1], self.node_map[gene2]
                edge_index.extend([[i, j], [j, i]])
                edge_weights.extend([weight, weight])

            temporal_graphs[t] = G
            temporal_edge_indices[t] = torch.tensor(edge_index).t().contiguous()
            temporal_edge_attrs[t] = torch.tensor(edge_weights, dtype=torch.float32).unsqueeze(1)
            
            # original embeddings for this time point
            print(f"Generating original (non-temporal) embeddings for time {t}")
            node2vec = Node2Vec(
                G,
                dimensions=self.embedding_dim // 2,  # Half the dimensions for original embeddings
                walk_length=25,
                num_walks=75,
                p=1.0,
                q=1.0,
                workers=1,
                seed=42
            )
            
            model = node2vec.fit(window=5, min_count=1, batch_words=4)
            
            gene_embeddings = {}
            for gene in self.node_map.keys():
                if gene in model.wv:
                    gene_embeddings[gene] = torch.tensor(model.wv[gene], dtype=torch.float32)
                else:
                    gene_embeddings[gene] = torch.zeros(self.embedding_dim // 2, dtype=torch.float32)
            
            original_embeddings[t] = gene_embeddings
        
        # temporal embeddings
        temporal_node2vec = TemporalNode2Vec(
            dimensions=self.embedding_dim // 2,  # Half the dimensions for temporal embeddings
            walk_length=25,
            num_walks=75,
            p=1.0,
            q=1.0,
            workers=1,
            seed=42,
            temporal_weight=0.5  
        )
        
        temporal_features_raw = temporal_node2vec.temporal_fit(
            temporal_graphs=temporal_graphs,
            time_points=self.time_points,
            node_map=self.node_map,
            window=5,
            min_count=1,
            batch_words=4
        )
        
        combined_features = {}
        for t in self.time_points:
            gene_features = []
            
            for i, gene in enumerate(self.node_map.keys()):
                orig_emb = original_embeddings[t][gene]
                
                temp_emb = temporal_features_raw[t][i]
               
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                            (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
                normalized_expr = (expr_value - global_min) / (global_max - global_min + 1e-8)
                
                combined_emb = torch.cat([orig_emb, temp_emb])
                
                gene_features.append(combined_emb)
            
            combined_features[t] = torch.stack(gene_features)
            
            print(f"\nCombined feature statistics for time {t}:")
            print(f"Shape: {combined_features[t].shape}")
            print(f"Min: {combined_features[t].min().item():.4f}, Max: {combined_features[t].max().item():.4f}")
            print(f"Mean: {combined_features[t].mean().item():.4f}, Std: {combined_features[t].std().item():.4f}")
        
        return combined_features, temporal_edge_indices, temporal_edge_attrs

    def get_edge_index_and_attr(self):
        """Convert base graph to PyG format"""
        edge_index = []
        edge_weights = []
        
        for u, v, d in self.base_graph.edges(data=True):
            i, j = self.node_map[u], self.node_map[v]
            edge_index.extend([[i, j], [j, i]])
            edge_weights.extend([d['weight'], d['weight']])
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        edge_weights = torch.tensor(edge_weights, dtype=torch.float32)
        edge_attr = edge_weights.unsqueeze(1)
        
        return edge_index, edge_attr
    
    def get_pyg_graph(self, time_point):
        """Create PyG graph for a specific time point"""
        return Data(
            x=self.node_features[time_point],
            edge_index=self.temporal_edge_indices[time_point],
            edge_attr=self.temporal_edge_attrs[time_point],
            num_nodes=self.num_nodes
        )
    
    def get_temporal_sequences(self):
        sequences = []
        labels = []
    
        #print("\nAvailable time points:", self.time_points)
        
        # First, create a clean dictionary of gene expressions across time
        gene_expressions = {}
        for t in self.time_points:
            # Convert time point to string for column access
            time_col = f'Gene1_Time_{t}'
            gene_expressions[t] = {}
            
            #print(f"\nProcessing time point {t}")  # Debug print
            
            for gene in self.node_map.keys():
                # Use the correct column name format
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                # Take the first non-empty value
                if len(gene1_expr) > 0:
                    expr_value = gene1_expr[0]
                elif len(gene2_expr) > 0:
                    expr_value = gene2_expr[0]
                else:
                    print(f"Warning: No expression found for gene {gene} at time {t}")
                    expr_value = 0.0
                    
                gene_expressions[t][gene] = expr_value

        print("\nExpression value check:")
        for t in self.time_points[:5]:  # First 5 time points
            print(f"\nTime point {t}:")
            for gene in list(self.node_map.keys())[:5]:  # First 5 genes
                print(f"Gene {gene}: {gene_expressions[t][gene]}")
        
        # Create sequences
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            input_times = self.time_points[i:i+self.seq_len]
            target_times = self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]
            
            print(f"\nSequence {i}:")
            print(f"Input times: {input_times}")
            print(f"Target times: {target_times}")
            
            # Create sequence graphs
            seq_graphs = []
            for t in input_times:
                # The features should already be 32-dimensional from Node2Vec
                features = torch.tensor([gene_expressions[t][gene] for gene in self.node_map.keys()], 
                                    dtype=torch.float32)
                graph = self.get_pyg_graph(t)
                # Node2Vec features are already shape [num_nodes, 32]
                #print(f"Graph features shape: {graph.x.shape}")  # Should be [52, 32]
                seq_graphs.append(graph)
            
            # Create label graphs
            label_graphs = []
            for t in target_times:
                graph = self.get_pyg_graph(t)
                label_graphs.append(graph)
            
            sequences.append(seq_graphs)
            labels.append(label_graphs)
        
        return sequences, labels
    
    def get_temporal_sequences_shuffle(self):
        sequences = []
        labels = []

        # First, create a clean dictionary of gene expressions across time
        gene_expressions = {}
        for t in self.time_points:
            # Convert time point to string for column access
            time_col = f'Gene1_Time_{t}'
            gene_expressions[t] = {}
            
            for gene in self.node_map.keys():
                # Use the correct column name format
                gene1_expr = self.df[self.df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
                gene2_expr = self.df[self.df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
                
                # Take the first non-empty value
                if len(gene1_expr) > 0:
                    expr_value = gene1_expr[0]
                elif len(gene2_expr) > 0:
                    expr_value = gene2_expr[0]
                else:
                    print(f"Warning: No expression found for gene {gene} at time {t}")
                    expr_value = 0.0
                    
                gene_expressions[t][gene] = expr_value

        print("\nExpression value check:")
        for t in self.time_points[:5]:  # First 5 time points
            print(f"\nTime point {t}:")
            for gene in list(self.node_map.keys())[:5]:  # First 5 genes
                print(f"Gene {gene}: {gene_expressions[t][gene]}")
        
        # Create sequences with indices to track order
        sequence_indices = []
        for i in range(len(self.time_points) - self.seq_len - self.pred_len + 1):
            input_times = self.time_points[i:i+self.seq_len]
            target_times = self.time_points[i+self.seq_len:i+self.seq_len+self.pred_len]
            
            print(f"\nSequence {i}:")
            print(f"Input times: {input_times}")
            print(f"Target times: {target_times}")
            
            seq_graphs = []
            for t in input_times:
                features = torch.tensor([gene_expressions[t][gene] for gene in self.node_map.keys()], 
                                    dtype=torch.float32)
                graph = self.get_pyg_graph(t)
                seq_graphs.append(graph)
            
            label_graphs = []
            for t in target_times:
                graph = self.get_pyg_graph(t)
                label_graphs.append(graph)
            
            sequences.append(seq_graphs)
            labels.append(label_graphs)
            sequence_indices.append(i) 
        
        random.seed(42)  
        shuffled_indices = sequence_indices.copy()
        random.shuffle(shuffled_indices)
        
        shuffled_sequences = []
        shuffled_labels = []
        for idx in shuffled_indices:
            shuffled_sequences.append(sequences[idx])
            shuffled_labels.append(labels[idx])
        
        print(f"\nOriginal sequence order: {sequence_indices}")
        print(f"Shuffled sequence order: {shuffled_indices}")
        
        return shuffled_sequences, shuffled_labels
    
    def split_sequences(self,sequences, labels):
        torch.manual_seed(42)
        
        n_samples = len(sequences)
        n_train = int(n_samples * (1 - 0.2))

        indices = torch.randperm(n_samples)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_sequences = [sequences[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        val_sequences = [sequences[i] for i in val_idx]
        val_labels = [labels[i] for i in val_idx]
        
        print("\nData Split Statistics:")
        print(f"Total sequences: {n_samples}")
        print(f"Training sequences: {len(train_sequences)} ({len(train_sequences)/n_samples:.1%})")
        print(f"Validation sequences: {len(val_sequences)} ({len(val_sequences)/n_samples:.1%})")
        
        return train_sequences, train_labels, val_sequences, val_labels, train_idx, val_idx
        
def visualize_enhanced_gene_graph(base_graph, gene_names, time_point):
    plt.figure(figsize=(18,16))

    pos = nx.spring_layout(base_graph, seed=42, k=2)

    node_labels = {i: gene for gene, i in gene_names.items() if i in base_graph.nodes()}
    nx.draw_networkx_nodes(base_graph, pos, node_size=800, node_color='skyblue', alpha=0.6)

    edge_weights = [base_graph[u][v]['weight'] for u,v in base_graph.edges()]
    edge_labels = {(u,v): f"{base_graph[u][v]['weight']:.2f}" for u,v in base_graph.edges()}

    nx.draw_networkx_edges(base_graph, pos, width=2.0, alpha=0.6, edge_color=edge_weights, edge_cmap=plt.cm.Blues_r)
    nx.draw_networkx_labels(base_graph, pos, labels=node_labels, font_size=11, font_weight='bold')
    nx.draw_networkx_edge_labels(base_graph, pos, edge_labels=edge_labels, font_size=12, font_color='black')

    plt.axis('off')
    plt.tight_layout()
    clean_time = str(time_point).replace('.', '_')
    plt.savefig(f"plottings_STGCN_clustered/graph_time_{clean_time}.png")
    plt.close()
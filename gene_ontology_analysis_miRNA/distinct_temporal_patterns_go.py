import pandas as pd
import numpy as np
from scipy.stats import zscore
from sklearn.cluster import KMeans
import pandas as pd
import requests
import json
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import linregress

ENRICHR_URL = "https://maayanlab.cloud/Enrichr"

def clean_gene_name(gene_name):
    if pd.isna(gene_name):
        return gene_name
    return gene_name.split('(')[0].strip()

def extract_expression_values(df, time_points):
    all_expressions = []
    expression_values = {}

    for gene in set(df['Gene1_clean']).union(set(df['Gene2_clean'])):
        gene_expressions = []
        for t in time_points:
            gene1_expr = df[df['Gene1_clean'] == gene][f'Gene1_Time_{t}'].values
            gene2_expr = df[df['Gene2_clean'] == gene][f'Gene2_Time_{t}'].values
            expr_value = gene1_expr[0] if len(gene1_expr) > 0 else \
                        (gene2_expr[0] if len(gene2_expr) > 0 else 0.0)
            gene_expressions.append(expr_value)
        
        expression_values[gene] = gene_expressions
        all_expressions.extend(gene_expressions)

    global_min, global_max = min(all_expressions), max(all_expressions) 
    for gene in expression_values:
        expression_values[gene] = [(x - global_min) / (global_max - global_min) for x in expression_values[gene]] # normalize expression values same as in embedding creation

    return expression_values

def identify_temporal_clusters(expression_values, n_clusters=3):

    genes = list(expression_values.keys())
    slopes = []

    for gene in genes:
        trend = expression_values[gene]
        slope, _, _, _, _ = linregress(range(len(trend)), trend)
        slopes.append(slope)

    for gene, slope in zip(genes, slopes):
        print(f"{gene}: {slope}")

    lower_threshold = np.percentile(slopes, 33)  # Bottom third
    upper_threshold = np.percentile(slopes, 67)  # Top third

    print(f"\nSlope Thresholds:\nLower: {lower_threshold}\nUpper: {upper_threshold}\n")

    clusters = {i: [] for i in range(n_clusters)}
    cluster_types = {}

    for gene, slope in zip(genes, slopes):
        if slope <= lower_threshold:
            clusters[0].append(gene)
        elif slope >= upper_threshold:
            clusters[2].append(gene)
        else:
            clusters[1].append(gene)

    cluster_types[0] = "Downregulated"
    cluster_types[1] = "Stable"
    cluster_types[2] = "Upregulated"

    for i in range(n_clusters):
        avg_slope = np.mean([slopes[genes.index(g)] for g in clusters[i]])
        print(f"Cluster {i} ({cluster_types[i]}) - Average slope: {avg_slope}")
        print(f"Genes in Cluster {i} ({cluster_types[i]}):")
        for gene in clusters[i]:
            print(f" - {gene}")

    return clusters, cluster_types


def get_enrichr_results(gene_list, database):
    genes_str = '\n'.join(gene_list)
    payload = {
        'list': (None, genes_str),
        'description': (None, 'Temporal Gene Cluster Analysis')
    }
    
    response = requests.post(f"{ENRICHR_URL}/addList", files=payload)
    if not response.ok:
        raise Exception("Error submitting gene list to Enrichr.")
    
    data = json.loads(response.text)
    user_list_id = data['userListId']
    
    response = requests.get(f"{ENRICHR_URL}/enrich",
                            params={'userListId': user_list_id, 'backgroundType': database})
    if not response.ok:
        raise Exception("Error retrieving enrichment results from Enrichr.")
    
    return pd.DataFrame(json.loads(response.text)[database], 
                        columns=['Rank', 'Term', 'P-value', 'Z-score', 'Combined score', 
                                 'Overlapping genes', 'Adjusted p-value', 'Old p-value', 
                                 'Old adjusted p-value'])

def analyze_clusters_with_go(clusters, databases=["GO_Biological_Process_2021", "GO_Molecular_Function_2021"]):
    if not os.path.exists("GO_results_temporal_patterns"):
        os.makedirs("GO_results_temporal_patterns")
    
    results = {}

    excel_filename = "temporal_go_analysis.xlsx"
    excel_file_pth = f"GO_results_temporal_patterns/{excel_filename}"
    with pd.ExcelWriter(excel_file_pth) as writer:
        for cluster_id, gene_list in clusters.items():
            print(f"\nRunning GO analysis for Cluster {cluster_id} ({len(gene_list)} genes)...")
            cluster_results = {}

            for db in databases:
                try:
                    go_results = get_enrichr_results(gene_list, db)
                    
                    if not go_results.empty:
                        sheet_name = f"Cluster_{cluster_id}_{db.split('_')[1]}"  # Shorter sheet name
                        go_results.to_excel(writer, sheet_name=sheet_name, index=False)
                        print(f"Saved GO results for Cluster {cluster_id} - {db} to Excel")

                    cluster_results[db] = go_results
                
                except Exception as e:
                    print(f"Error analyzing Cluster {cluster_id} in {db}: {e}")
                    cluster_results[db] = pd.DataFrame()  
            
            results[cluster_id] = cluster_results
    
    print(f"\nGO results saved in '{excel_file_pth}'")
    return results

def run_temporal_clustering_and_go_analysis(expression_data_csv, num_clusters=3, expression_values=None):
    if expression_values is None:
        raise ValueError("Expression values must be provided!")

    print("\nPerforming Temporal Clustering...")
    clusters, cluster_types = identify_temporal_clusters(expression_values, num_clusters)
    print(f"Cluster types: {cluster_types}")

    for cluster, genes in clusters.items():
        print(f"\nCluster {cluster} : {len(genes)} genes")
        #print(", ".join(genes[:10]) + ("..." if len(genes) > 10 else ""))
        print(", ".join(genes)) 

    print("\n🔍 Running GO Enrichment Analysis...")
    go_results = analyze_clusters_with_go(clusters)

    print("\nTemporal clustering & GO enrichment analysis completed!")
    print("Results saved in 'temporal_go_analysis.xlsx'")

if __name__ == "__main__":

    #csv_file = "/Users/beyzakaya/Desktop/temporal gene/mapped/enhanced_interactions_synthetic_simple.csv"
    csv_file = '/Users/beyzakaya/Desktop/temporalHiC/mapped/miRNA_expression_mean/standardized_time_columns_meaned_expression_values_get_closest.csv'
    df = pd.read_csv(csv_file)
    #df['Gene1_clean'] = df['Gene1'].apply(clean_gene_name)
    #df['Gene2_clean'] = df['Gene2'].apply(clean_gene_name)

    df['Gene1_clean'] = df['Gene1']
    df['Gene2_clean'] = df['Gene2']

    #time_points = [float(col.split('_')[-1]) for col in df.columns if "Gene1_Time" in col]
    time_cols = [col for col in df.columns if 'Time_' in col]
    time_points = sorted(list(set([float(col.split('_')[-1])  
                                          for col in time_cols if 'Gene1' in col])))
    time_points = [float(tp) for tp in time_points]
    
    time_points = [tp for tp in time_points if tp != 154.0]
    df = df.loc[:, ~df.columns.str.contains('Time_154.0', case=False)]
    print(f"Time points for analyzes: {time_points}")
    expression_values = extract_expression_values(df, time_points)
    clusters, cluster_types = identify_temporal_clusters(expression_values)

    #run_temporal_clustering_and_go_analysis(csv_file, num_clusters=3, expression_values=expression_values)

    

import pandas as pd
import numpy as np
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import community
import matplotlib.cm as cm
import requests

df = pd.read_csv("/Users/beyzakaya/Desktop/temporal gene/mapped/enhanced_interactions_synthetic_simple.csv")

time_columns = [col for col in df.columns if "Time_" in col]
expression_data = df[["Gene1"] + time_columns].rename(columns={"Gene1": "Gene"}).set_index("Gene")
print(expression_data.index.duplicated().sum(), "duplicate gene names found!")
expression_data = expression_data.groupby(expression_data.index).mean()

corr_matrix = expression_data.T.corr(method="pearson") 

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0, annot=False)
plt.title("Gene Co-Expression Correlation Matrix")
plt.savefig('coexpressin_correl_matrix.png')
plt.show()


threshold = 0.7
edges = [
    (gene1, gene2, corr_matrix.at[gene1, gene2])  
    for gene1 in corr_matrix.index 
    for gene2 in corr_matrix.columns 
    if gene1 != gene2 and abs(corr_matrix.at[gene1, gene2]) > threshold
]


G = nx.Graph()
G.add_weighted_edges_from(edges)

plt.figure(figsize=(12, 10))
pos = nx.spring_layout(G, seed=42)  
nx.draw(G, pos, with_labels=True, node_size=500, edge_color="gray", font_size=10)
plt.title("Gene Co-Expression Network")
plt.savefig('coexpressin_network.png')
plt.show()


communities = community.louvain_communities(G)

num_communities = len(communities)
colors = cm.viridis(np.linspace(0, 1, num_communities))

community_color_map = {}
for i, c in enumerate(communities):
    for gene in c:
        community_color_map[gene] = colors[i]

node_sizes = [G.degree(n) * 100 for n in G.nodes()]

plt.figure(figsize=(14, 12))
pos = nx.spring_layout(G, seed=42, k=0.3)  

nx.draw_networkx_nodes(G, pos, node_color=[community_color_map[n] for n in G.nodes()],
                       node_size=node_sizes, alpha=0.8, cmap=cm.viridis)

nx.draw_networkx_edges(G, pos, alpha=0.3, edge_color="gray")

hub_genes = [n for n in G.nodes() if G.degree(n) > 3]
nx.draw_networkx_labels(G, pos, labels={n: n for n in hub_genes}, font_size=12)

plt.title("Gene Co-Expression Network with Community Detection", fontsize=16)
plt.axis('off')
plt.savefig('coexpressin_network_community_detection.png')
plt.show()

degree_dict = dict(G.degree())
sorted_hubs = sorted(degree_dict.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top Hub Genes:", sorted_hubs)

for gene in [h[0] for h in sorted_hubs]:
    response = requests.get(f"https://string-db.org/api/tsv/network?identifiers={gene}&species=9606")
    print(f"Protein Interactions for {gene}:\n", response.text)

for i, c in enumerate(communities):
    print(f"Cluster {i+1}: {c}")

def analyze_chromosomal_distribution(df):

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x='Gene1_Chromosome')
    plt.title('Gene1 Chromosome Distribution')
    plt.xticks(rotation=45)
    
    plt.subplot(1, 2, 2)
    sns.countplot(data=df, x='Gene2_Chromosome')
    plt.title('Gene2 Chromosome Distribution')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()
    
    same_chrom = df['Gene1_Chromosome'] == df['Gene2_Chromosome']
    print("\nInteraction Types:")
    print(f"Intra-chromosomal: {sum(same_chrom)}")
    print(f"Inter-chromosomal: {sum(~same_chrom)}")

def analyze_time_series(df):

    time_cols = [col for col in df.columns if 'Time' in col]
    time_points = sorted(list(set([float(col.split('_')[-1]) 
                                 for col in time_cols])))
    
    plt.figure(figsize=(12, 6))
    
    gene1_means = []
    gene2_means = []
    for time in time_points:
        gene1_col = f'Gene1_Time_{time}'
        gene2_col = f'Gene2_Time_{time}'
        gene1_means.append(df[gene1_col].mean())
        gene2_means.append(df[gene2_col].mean())
    
    plt.plot(time_points, gene1_means, 'o-', label='Gene1')
    plt.plot(time_points, gene2_means, 'o-', label='Gene2')
    plt.xlabel('Time')
    plt.ylabel('Average Expression')
    plt.title('Expression Patterns Over Time')
    plt.legend()
    plt.show()


print("Analyzing chromosomal distribution...")
analyze_chromosomal_distribution(df)

print("\nAnalyzing time series patterns...")
analyze_time_series(df)



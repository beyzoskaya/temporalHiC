import pandas as pd
import gseapy as gp
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('/Users/beyzakaya/Desktop/temporalHiC/mapped/mRNA/enhanced_interactions_synthetic_simple_mRNA.csv')
expression_columns = [col for col in df.columns if 'Time' in col]

unique_genes = pd.concat([df['Gene1'], df['Gene2']]).unique()
unique_genes = [str(gene).upper() for gene in unique_genes]
#unique_genes = [gene.replace('INTEGRIN SUBUNIT ALPHA 8', 'ITGA8') for gene in unique_genes]

expression_matrix = pd.DataFrame(index=unique_genes)

# Each time column corresponds to the gene expression at different time points
for time_col in expression_columns:
    gene1_values = df.set_index('Gene1')[time_col]
    gene2_values = df.set_index('Gene2')[time_col]
    
    combined_values = pd.concat([gene1_values, gene2_values])
    mean_values = combined_values.groupby(combined_values.index.str.upper()).mean()
    
    expression_matrix[time_col] = mean_values

expression_matrix = expression_matrix.fillna(0)

early_time_points = [col for col in expression_columns if float(col.split('_')[-1]) <= 13.0]
late_time_points = [col for col in expression_columns if float(col.split('_')[-1]) > 13.0]

cls = ['Early'] * len(early_time_points) + ['Late'] * len(late_time_points)

gene_list = [
    "HIST1H1B", "VIM", "P-63", "INMT", "ADAMTSL2", "TNC", "FGF18", "SHISA3", 
    "INTEGRIN SUBUNIT ALPHA 8", "HIST1H2AB", "CD38", "MMP-3", "LRP2", "PPIA", 
    "THTPA", "VEGF", "GATA-6", "ABCA3", "KCNMA1", "TFRC", "RAGE", "F13A1", "MCPt4",
    "FOXF2", "EPHA7", "AGER", "HMBS", "E2F8", "TGFB1", "TTF-1", "CLAUDIN5", "GUCY1A2  SGC", 
    "PRIM2", "TBP", "SFTP-D", "N-CADHERIN", "THY1", "CLAUDIN 1", "IGFBP3", "EGFR", "YWHAZ", 
    "HPRT", "ABCD1", "NME3", "MGAT4A", "MMP7", "HPGDS", "ABCG2", "AMACR"
]

gene_list_mirna = [
    '1700025G04Rik', '3425401B19Rik', 'STUM', 'Abcg1', 'Abra', 'Abt1', 
    'Acp1', 'Adora1', 'Aff4', 'Amotl1', 'Amph', 'Ankrd28', 'Ankrd33b', 
    'Aptx', 'Arhgef4', 'Atp13a3', 'Bach2', 'Bcl6', 'CDR1', 'CTNND1', 
    'Odad1', 'Ccdc88a', 'Ccl21d', 'Cdh12', 'Cdk19', 'Cdk6', 'Celf2', 
    'Chn1', 'Clcn3', 'Col5a3', 'Colq', 'Cpeb2', 'Cplx2', 'Crybg1', 
    'Ctnnd1', 'Cyp2b13', 'Cyp2b9', 'Cyp7a1', 'Macir', 'Dap', 
    'Dolpp1', 'Dpp10', 'Dusp23', 'Dyrk1b', 'Ebf1', 'Ets1', 'Ewsr1',
    'Fam114a1', 'Fam163b', 'Fam171a1', 'LRATD1', 'CIBAR1', 'Fbxl12os',
    'Fgf21', 'Fli1', 'Flrt1', 'Fzd7', 'G2e3', 'Gabra1', 'Gm16387', 'Gm16485',
    'DYNLT2A2', 'Gm7293', 'Gmpr', 'Gng12', 'Gnl1', 'Gpm6a', 'Gpr158', 'Gramd2a', 
    'Hbq1a', 'H4c2', 'Hnrnph1', 'Hnrnpu', 'Ier5l', 'Ikzf2', 'Ing2', 'Inhbb', 'Ino80d', 
    'Itk', 'Jcad', 'Kbtbd3', 'Kcnh7', 'Kctd16', 'Klf7', 'Klhdc8a', 'Klhl20', 'Lin28b', 
    'Ly96', 'MYB', 'Map3k2', 'Map4k4', 'Mars2', 'Mdm4', 'Mmp16', 'Mrrf', 'Mzf1', 
    'Nlgn1', 'Nlrc3', 'Nmi', 'Nr6a1', 'Nsmf', 'Nxph1', 'Pakap', 'Pappa', 'Peg10',
    'Pes1', 'Pfkfb2', 'Pfn4', 'Phlpp1', 'Plekhb2', 'Plxdc2', 'Ppp2ca', 'Proz', 'Prx',
    'Ptchd4', 'Rbm47', 'Rbpj', 'Rnf152', 'SLY', 'Satb2', 'Scaf8', 'Sftpb', 'Skida1', 
    'Slc2a8', 'Slc8a2', 'Sox4', 'Strn3', 'Tagap1', 'Tbc1d24', 'Tbc1d8', 'Tbp', 'Thnsl1', 
    'Thsd1', 'Tmem245', 'Tmem38b', 'Tpsb2', 'Uap1l1', 'Ube2e3', 'Wdr12', 'Zbtb41', 
    'Zfp467', 'Zfp607a', 'Zfp648', 'Ppif', 'BLTP3A', 'Ttc33', 'Wdr37', 'Ywhaq', 'Trim7', 'Zfp846', 
    'Zfp292', 'Zbtb20', 'Slc9a6', 'Map3k20', 'Zfp850', 'Plagl1', 'Tbc1d1', 'Tusc3', 'Zfand5', 
    'Zfp800', 'Pcdhb22', 'Tsc22d2'
]
gene_list = [gene.upper() for gene in gene_list]

try:
    # Gene set enrichment analysis
    gsea = gp.gsea(
        data=expression_matrix[early_time_points + late_time_points],
        gene_sets={'my_gene_set': gene_list},
        cls=cls,
        method='t_test', # rank for this method
        permutation_type='phenotype',
        number_of_permutations=1000,
        verbose=True
    )
    
    gsea_results = gsea.res2d
    print("\nGSEA Results:")
    print(gsea_results)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 1, 1)
    result_metrics = pd.DataFrame({
        'Metric': ['Enrichment Score (ES)', 'Normalized ES', 'Nominal p-value', 'FDR q-value'],
        'Value': [
            gsea_results.loc[0, 'ES'],
            gsea_results.loc[0, 'NES'],
            gsea_results.loc[0, 'NOM p-val'],
            gsea_results.loc[0, 'FDR q-val']
        ]
    })
    
    sns.barplot(x='Metric', y='Value', data=result_metrics)
    plt.xticks(rotation=45)
    plt.title('GSEA Metrics')
    
    plt.subplot(2, 1, 2)
    # leading edge genes refer to the subset of genes that are most responsible for the enrichment
    # leading genes are usually the ones that show the most significant association with the phenotype or condition
    leading_genes = gsea_results.loc[0, 'Lead_genes'].split(';')
    leading_gene_values = pd.DataFrame({
        'Gene': leading_genes,
        'Early_Mean': expression_matrix[early_time_points].loc[leading_genes].mean(axis=1),
        'Late_Mean': expression_matrix[late_time_points].loc[leading_genes].mean(axis=1)
    })
   
    leading_gene_matrix = leading_gene_values[['Early_Mean', 'Late_Mean']]
    sns.heatmap(leading_gene_matrix, 
                annot=True, 
                fmt='.2f',
                cmap='RdBu_r',
                yticklabels=leading_genes)
    plt.title('Expression Pattern of Leading Edge Genes')
    
    plt.tight_layout()
    plt.savefig('gsea_plottings_mirna/expr_patterns.png')
    plt.show()
    
    print("\nDetailed GSEA Analysis:")
    print(f"Enrichment Score (ES): {gsea_results.loc[0, 'ES']:.3f}")
    print(f"Normalized ES: {gsea_results.loc[0, 'NES']:.3f}")
    print(f"Nominal p-value: {gsea_results.loc[0, 'NOM p-val']:.3f}")
    print(f"FDR q-value: {gsea_results.loc[0, 'FDR q-val']:.3f}")
    print("\nLeading Edge Genes:")
    for gene in leading_genes:
        early_mean = expression_matrix[early_time_points].loc[gene].mean()
        late_mean = expression_matrix[late_time_points].loc[gene].mean()
        print(f"{gene}: Early mean = {early_mean:.2f}, Late mean = {late_mean:.2f}")
    
except Exception as e:
    print(f"Error during GSEA analysis: {str(e)}")
    print("\nShape of expression matrix:", expression_matrix.shape)
    print("Number of class labels:", len(cls))
    print("Sample of expression matrix:")
    print(expression_matrix.head())

def create_additional_gsea_visualizations(gsea, expression_matrix, early_time_points, late_time_points):
    gsea_results = gsea.res2d
    leading_genes = gsea_results.loc[0, 'Lead_genes'].split(';')

    # 1. Time Series Plot for Leading Edge Genes
    plt.figure(figsize=(15, 6))
    time_points = [float(col.split('_')[-1]) for col in early_time_points + late_time_points]
    all_timepoints = early_time_points + late_time_points
    
    for gene in leading_genes:
        gene_values = expression_matrix.loc[gene, all_timepoints].values
        plt.plot(time_points, gene_values, marker='o', label=gene)
    
    plt.axvline(x=4.0, color='r', linestyle='--', label='Early/Late Boundary')
    plt.xlabel('Time Points')
    plt.ylabel('Expression Values')
    plt.title('Expression Trajectories of Leading Edge Genes')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('gsea_plottings_mirna/expr_trajectories.png')

    print(expression_matrix.index)
    custom_genes = ['VIM', 'INTEGRIN SUBUNIT ALPHA 8', 'HPRT', 'ADAMTSL2', 'TTF-1', 'INMT', 'CLAUDIN5', 'ABCD1']
    print(expression_matrix.loc['INTEGRIN SUBUNIT ALPHA 8'])

    valid_custom_genes = [gene for gene in custom_genes if gene in expression_matrix.index]
    print(f"Valid genes: {valid_custom_genes}")

    selected_expr = expression_matrix.loc[custom_genes, early_time_points + late_time_points]

    correlation_matrix = selected_expr.T.corr()

    gene_name_mapping = {
    'INTEGRIN SUBUNIT ALPHA 8': 'INTEGRIN8',  
    }

    correlation_matrix = correlation_matrix.rename(index=gene_name_mapping, columns=gene_name_mapping)

    # Plot the heatmap
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0,
                cbar_kws={"label": "Pearson Correlation"})

    # This line controls the y-axis label rotation
    ax.set_yticklabels(ax.get_yticklabels(), rotation=90)

    # Optionally, if you want to adjust x-axis labels too
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    plt.title("Correlation Matrix for Top 8 Performed Genes")
    plt.tight_layout()
    plt.savefig("custom_genes_correlation.png")
    plt.show()
    
    # 2. Boxplot Comparison
    plt.figure(figsize=(12, 6))
    early_data = []
    late_data = []
    gene_labels = []
    
    for gene in leading_genes:
        early_values = expression_matrix.loc[gene, early_time_points].values
        late_values = expression_matrix.loc[gene, late_time_points].values
        early_data.extend(early_values)
        late_data.extend(late_values)
        gene_labels.extend([gene] * len(early_values))
        gene_labels.extend([gene] * len(late_values))
    
    comparison_df = pd.DataFrame({
        'Expression': early_data + late_data,
        'Time': ['Early'] * len(early_data) + ['Late'] * len(late_data),
        'Gene': gene_labels
    })
    
    sns.boxplot(x='Gene', y='Expression', hue='Time', data=comparison_df)
    plt.xticks(rotation=45)
    plt.title('Early vs Late Expression Distribution')
    plt.tight_layout()
    plt.savefig('gsea_plottings_mirna/early_late_distribution.png')
    
    # 3. Correlation Heatmap
    # high positive correlation means that the gene expressions of these two genes are highly similar. These genes show similar expression patterns across time points
    # This could indicate that these genes are co-regulated or part of the same biological pathway or process. 
    # correlation closer to -1 indicates an inverse relationship. That is, when the expression of one gene goes up, the expression of the other gene goes down, and vice versa.
    plt.figure(figsize=(10, 8))
    gene_correlations = expression_matrix.loc[leading_genes, all_timepoints].T.corr()
    sns.heatmap(gene_correlations, 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm',
                center=0)
    plt.title('Correlation Between Leading Edge Genes')
    plt.tight_layout()
    plt.savefig('gsea_plottings_mirna/correl_between_edge_genes.png')
    
    # 4. Expression Change Plot
    plt.figure(figsize=(10, 6))
    early_means = expression_matrix.loc[leading_genes, early_time_points].mean(axis=1)
    late_means = expression_matrix.loc[leading_genes, late_time_points].mean(axis=1)
    fold_changes = late_means - early_means
    
    sns.barplot(x=leading_genes, y=fold_changes)
    plt.xticks(rotation=45)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('Expression Changes (Late - Early)')
    plt.ylabel('Log2 Expression Change')
    plt.tight_layout()
    plt.savefig('gsea_plottings_mirna/expr_changes.png')
    
    return plt.gcf()

create_additional_gsea_visualizations(gsea, expression_matrix, early_time_points, late_time_points)
plt.show()
import pandas as pd
import requests
import json
import os
import logging
from distinct_temporal_patterns_go import clean_gene_name
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ENRICHR_URL = "https://maayanlab.cloud/Enrichr"
#DATABASES = ["Mouse_Gene_Atlas", "ARCHS4_Tissues"]
DATABASES = ["Mouse_Gene_Atlas"]

def check_gene_list(gene_list):
    logger.info("\nAnalyzing gene list:")
    logger.info(f"Total number of genes: {len(gene_list)}")
    
    case_types = {
        'uppercase': len([g for g in gene_list if g.isupper()]),
        'lowercase': len([g for g in gene_list if g.islower()]),
        'mixed': len([g for g in gene_list if not g.isupper() and not g.islower()])
    }
    
    logger.info("\nCase analysis:")
    for case_type, count in case_types.items():
        logger.info(f"{case_type}: {count} genes")
    
    logger.info("\nSample of genes (first 10):")
    for gene in gene_list[:10]:
        logger.info(f"Gene: {gene}")

def get_enrichr_results(gene_list, database):
    gene_list = [str(gene).upper() for gene in gene_list if gene and pd.notna(gene)]
    genes_str = '\n'.join(gene_list)
    
    try:
        logger.info(f"\nSubmitting {len(gene_list)} genes to Enrichr")
        
        payload = {
            'list': (None, genes_str),
            'description': (None, 'Mouse genes tissue analysis')
        }
        
        response = requests.post(f"{ENRICHR_URL}/addList", files=payload)
        response.raise_for_status()
        submission_data = response.json()
        
        logger.info(f"Genes successfully submitted: {submission_data.get('shortId')}")
        
        if 'mapping' in submission_data:
            mapped_genes = len([g for g in submission_data['mapping'] if submission_data['mapping'][g]])
            logger.info(f"Successfully mapped genes: {mapped_genes}/{len(gene_list)}")
        
        user_list_id = submission_data['userListId']
        
        response = requests.get(
            f"{ENRICHR_URL}/enrich",
            params={'userListId': user_list_id, 'backgroundType': database}
        )
        response.raise_for_status()
        
        results = pd.DataFrame(
            response.json()[database],
            columns=['Rank', 'Term', 'P-value', 'Z-score', 'Combined score',
                    'Overlapping genes', 'Adjusted p-value', 'Old p-value',
                    'Old adjusted p-value']
        )
        
        if not results.empty:
            logger.info(f"\nEnrichment results summary for {database}:")
            logger.info(f"Total terms found: {len(results)}")
            logger.info(f"Terms with p < 0.05: {len(results[results['P-value'] < 0.05])}")
            logger.info(f"Terms with adjusted p < 0.05: {len(results[results['Adjusted p-value'] < 0.05])}")
        
        return results
    
    except Exception as e:
        logger.error(f"Error in enrichment analysis: {str(e)}")
        return pd.DataFrame()

def analyze_tissue_specificity(gene_list):
    check_gene_list(gene_list)
    
    os.makedirs("GO_results_tissue", exist_ok=True)
    output_file = "GO_results_tissue/tissue_enrichment_results.xlsx"
    
    try:
        with pd.ExcelWriter(output_file) as writer:
            for database in DATABASES:
                logger.info(f"\nAnalyzing with {database}...")
                
                results = get_enrichr_results(gene_list, database)
                
                if not results.empty:
                    results.to_excel(writer, sheet_name=f"{database[:31]}", index=False)

                    significant_results = results[results['Adjusted p-value'] < 0.05]
                    if not significant_results.empty:
                        logger.info(f"\nTop enriched terms in {database}:")
                        for _, row in significant_results.head().iterrows():
                            logger.info(f"Term: {row['Term']}")
                            logger.info(f"  Adj. p-value: {row['Adjusted p-value']:.2e}")
                            logger.info(f"  Overlapping genes: {row['Overlapping genes']}")
                    else:
                        logger.info(f"No significant terms found in {database}")
                else:
                    logger.info(f"No results returned for {database}")
        
        logger.info(f"\nResults saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

def plot_tissue_specific_heatmap(results, tissue_column='Term'):
    try:
        logger.info("Plotting tissue-specific enrichment heatmap...")
        
        tissue_data = results[['Term', 'P-value', 'Adjusted p-value', 'Z-score', 'Combined score']].sort_values(by='P-value')
        tissue_data = tissue_data.set_index(tissue_column)
        
        plt.figure(figsize=(20, 16))
        sns.heatmap(tissue_data[['Z-score', 'Combined score']], annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
        plt.title(f"Tissue-Specific Enrichment (Z-score & Combined score)")
        plt.ylabel('Tissue')
        plt.xlabel('Scores')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('GO_results_tissue/go_terms_tissue_specific_heatmap.png')
        plt.show()
        
    except Exception as e:
        logger.error(f"Error generating heatmap: {str(e)}")

def plot_enrichment_barplot(results, top_n=10):
    try:
        logger.info("Plotting bar plot for top enriched terms...")
        
        top_results = results.sort_values(by='Adjusted p-value').head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Adjusted p-value', y='Term', data=top_results, palette='viridis')
        plt.title(f"Top {top_n} Enriched Terms")
        plt.xlabel('Adjusted p-value')
        plt.ylabel('Enriched Terms')
        plt.tight_layout()
        plt.savefig('GO_results_tissue/go_terms_tissue.png')
        plt.show()
        
    except Exception as e:
        logger.error(f"Error generating bar plot: {str(e)}")

def analyze_tissue_specificity_with_plottings(gene_list):
    check_gene_list(gene_list)
    
    os.makedirs("GO_results_tissue", exist_ok=True)
    output_file = "GO_results_tissue/tissue_enrichment_results.xlsx"
    
    try:
        with pd.ExcelWriter(output_file) as writer:
            for database in DATABASES:
                logger.info(f"\nAnalyzing with {database}...")
                
                results = get_enrichr_results(gene_list, database)
                
                if not results.empty:
                    results.to_excel(writer, sheet_name=f"{database[:31]}", index=False)

                    plot_tissue_specific_heatmap(results)
                    plot_enrichment_barplot(results)
                    
                    significant_results = results[results['Adjusted p-value'] < 0.05]
                    if not significant_results.empty:
                        logger.info(f"\nTop enriched terms in {database}:")
                        for _, row in significant_results.head().iterrows():
                            logger.info(f"Term: {row['Term']}")
                            logger.info(f"  Adj. p-value: {row['Adjusted p-value']:.2e}")
                            logger.info(f"  Overlapping genes: {row['Overlapping genes']}")
                    else:
                        logger.info(f"No significant terms found in {database}")
                else:
                    logger.info(f"No results returned for {database}")
        
        logger.info(f"\nResults saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")

if __name__ == "__main__":
    try:
        #csv_file = "/Users/beyzakaya/Desktop/temporal gene/mapped/enhanced_interactions_synthetic_simple.csv" 
        csv_file = '/Users/beyzakaya/Desktop/temporalHiC/mapped/miRNA_expression_mean/standardized_time_columns_meaned_expression_values_get_closest.csv'
        df = pd.read_csv(csv_file)
        gene_list = pd.concat([df['Gene1'], df['Gene2']]).unique().tolist()

        #df['Gene1_clean'] = df['Gene1'].apply(clean_gene_name)
        #df['Gene2_clean'] = df['Gene2'].apply(clean_gene_name)
        
        gene_list = pd.concat([df['Gene1'], df['Gene2']]).unique().tolist()
        
        logger.info(f"Analyzing {len(gene_list)} genes...")
        
        analyze_tissue_specificity_with_plottings(gene_list)
        
    except FileNotFoundError:
        logger.error(f"Could not find file: {csv_file}")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
    
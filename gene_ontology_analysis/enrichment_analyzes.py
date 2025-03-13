from goatools import obo_parser
from goatools.go_enrichment import GOEnrichmentStudy
from goatools.anno.genetogo_reader import Gene2GoReader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import wget
import os
import gzip
import shutil

def download_go_database():
    if not os.path.exists('go-basic.obo'):
        print("\nDownloading GO database file...")
        url = "http://purl.obolibrary.org/obo/go/go-basic.obo"
        wget.download(url, "go-basic.obo")
        print("\nGO database downloaded successfully")

def download_gene2go():
    if not os.path.exists('gene2go'):
        print("\nDownloading gene2go data from NCBI...")
        url = "https://ftp.ncbi.nlm.nih.gov/gene/DATA/gene2go.gz"
        wget.download(url, "gene2go.gz")
        print("\nExtracting gene2go file...")
        
        with gzip.open('gene2go.gz', 'rb') as f_in:
            with open('gene2go', 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        
        os.remove('gene2go.gz')
        print("Gene2go file prepared successfully")

def run_go_analysis(study_genes, background_genes, obo_file="go-basic.obo"):
    download_go_database()
    download_gene2go()
    
    print("\nLoading GO database...")
    go = obo_parser.GODag(obo_file)
    
    print("Reading gene associations...")
    objanno = Gene2GoReader(filename='gene2go', taxids=[10090])
    ns2assoc = objanno.get_ns2assc()
    
    study_genes = [str(gene) for gene in study_genes]
    background_genes = [str(gene) for gene in background_genes]
    
    results_all = []
    
    for namespace, assoc in ns2assoc.items():
        print(f"\nAnalyzing {namespace} terms...")
        goeaobj = GOEnrichmentStudy(
            background_genes,
            assoc,
            go,
            propagate_counts=True,
            alpha=0.05,  # Significance threshold
            methods=['fdr_bh']  # Multiple testing correction
        )
        
        go_results = goeaobj.run_study(study_genes)
        
        if go_results:
            df = pd.DataFrame([
                {'GO_term': r.name,
                 'GO_id': r.GO,
                 'namespace': namespace,
                 'p_value': r.p_uncorrected,
                 'p_adj': r.p_fdr_bh,
                 'study_count': r.study_count,
                 'study_n': r.study_n,
                 'background_count': r.pop_count,
                 'background_n': r.pop_n,
                 'enrichment_ratio': (r.study_count/r.study_n)/(r.pop_count/r.pop_n),
                 'genes': ', '.join(r.study_items)
                } for r in go_results if hasattr(r, 'p_fdr_bh') and r.p_fdr_bh is not None
            ])
            results_all.append(df)
    
    if results_all:
        final_results = pd.concat(results_all, ignore_index=True)
        return final_results.sort_values('p_adj')
    else:
        return pd.DataFrame()

def visualize_results(results):
    """
    Create visualization of GO enrichment results
    """
    if results.empty:
        print("No significant GO terms found to visualize")
        return
    
    # Create scatter plot
    plt.figure(figsize=(12, 6))
    sns.scatterplot(
        data=results,
        x='enrichment_ratio',
        y='-log10(p_adj)',
        hue='namespace',
        size='study_count',
        sizes=(50, 400),
        alpha=0.6
    )
    
    # Add labels for significant terms
    significant_terms = results[results['p_adj'] < 0.05]
    for _, row in significant_terms.iterrows():
        plt.annotate(
            row['GO_term'][:30] + '...' if len(row['GO_term']) > 30 else row['GO_term'],
            (row['enrichment_ratio'], -np.log10(row['p_adj'])),
            xytext=(5, 5), textcoords='offset points',
            fontsize=8
        )
    
    plt.title('GO Term Enrichment Analysis')
    plt.xlabel('Enrichment Ratio')
    plt.ylabel('-log10(Adjusted P-value)')
    plt.legend(title='Namespace', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Study genes (your negatively correlated genes)
    study_genes = [26357, 17117]  # ABCG2 and AMACR
    
    # Background genes (all genes)
    background_genes = [
        21374, 14465, 12741, 26357, 14172, 16531, 241226, 56702, 22352, 21869,
        15288, 17117, 21803, 74145, 22339, 269181, 12494, 17227, 17392, 12737,
        54486, 108961, 79059, 234889, 21743, 13649, 77794, 14238, 16009, 22042,
        22631, 15452, 19076, 17393, 21838, 22061, 11666, 13841, 12558, 11596,
        20390, 21924
    ]
    
    try:
        print("Starting GO enrichment analysis...")
        results = run_go_analysis(study_genes, background_genes)
        
        if not results.empty:
            results.to_csv('go_enrichment_results.csv', index=False)
            print("\nResults saved to 'go_enrichment_results.csv'")
            
            print("\nTop enriched GO terms:")
            display_cols = ['GO_term', 'namespace', 'p_adj', 'enrichment_ratio', 'genes']
            pd.set_option('display.max_colwidth', None)
            print(results[display_cols].head(10))
            
            # Visualize results
            print("\nCreating visualization...")
            visualize_results(results)
        else:
            print("\nNo significant GO terms found")
            
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
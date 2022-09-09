import numpy as np
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns


def do_go_enrichment(gene_names, no_evidences=False, **kwargs):
    kwargs['no_evidences'] = no_evidences
    
    go = sc.queries.enrich(gene_names, 
                           gprofiler_kwargs=kwargs)
    # print(go.shape)
    go['neg_log10_p_value'] = -np.log10(go.p_value)
    return go

def plot_go_enrichment(go):
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(y='name', x='neg_log10_p_value', data=go.head(20), color='grey', ax=ax)
    sns.despine()

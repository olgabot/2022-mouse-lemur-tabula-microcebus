import scanpy as sc

def do_differential_expression(adata, groupby, method='logreg', use_raw=True,
                               penalty='l1', solver='saga', class_weight='balanced', n_jobs=16,
                               **kwargs):
    sc.tl.rank_genes_groups(adata, groupby=groupby,
                            method=method, use_raw=use_raw,
                            penalty=penalty, solver=solver,
                            class_weight=class_weight, **kwargs)
    sc.pl.rank_genes_groups(adata)
    

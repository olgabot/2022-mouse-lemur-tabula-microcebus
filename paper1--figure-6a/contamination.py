from typing import Union, Optional, Tuple, Collection, Sequence, Iterable


import numpy as np
import scanpy 
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns    

import anndata
from anndata import AnnData
from scanpy import logging as logg
from scanpy.preprocessing._distributed import materialize_as_ndarray
from scipy.sparse import issparse, isspmatrix_csr, csr_matrix, spmatrix




import _highly_variable_genes


anndata.__version__

scanpy.settings.verbosity = 3
scanpy.logging.print_versions()

scanpy.set_figure_params(frameon=False, color_map='magma_r')

# https://github.com/theislab/scanpy/blob/master/scanpy/plotting/_tools/scatterplots.py#L189
MARKER_SIZE_FACTOR = 120000

def subset_possibly_contaminating(adata, tissue, celltype, score_column, 
                                  col='cell_ontology_class'):
    # Get original marker size from scanpy's scatterplots:
    marker_size = MARKER_SIZE_FACTOR / adata.shape[0]
    
    is_celltype = adata.obs[col] == celltype
    print(f"is_celltype.sum(): {is_celltype.sum()}")
    
    non_tissue_gr0_bool = adata.obs.loc[adata.obs.tissue != tissue, score_column] > 0
    print(f"non_tissue_gr0_bool.sum(): {non_tissue_gr0_bool.sum()}")

    possibly_contaminating_cells = non_tissue_gr0_bool[non_tissue_gr0_bool].index
    print(f"len(possibly_contaminating_cells): {len(possibly_contaminating_cells)}")

    celltype_or_contaminating_cells = (adata.obs[col] == celltype) | adata.obs.index.isin(possibly_contaminating_cells)
    print(f"celltype_or_contaminating_cells.sum(): {celltype_or_contaminating_cells.sum()}")

    adata_subset = adata[celltype_or_contaminating_cells, :]
    print(f"adata_possibly_contaminated: {adata_subset}")
    # Plot UMAP using original coordinates
    sc.pl.umap(adata_subset, color='tissue', size=marker_size)
    sc.pl.umap(adata_subset, color='tissue')
    
    adata = compute_plot_pca_umap(adata, palette=None)

    # Plot PCA with cleaned free annotation
    sc.pl.umap(adata_subset, color=col)
    sc.pl.umap(adata_subset, color='n_counts', cmap='viridis')

    differential_expression(adata_subset, celltype, col=col)
    return adata_subset


def compute_plot_pca_umap(adata, palette=None, highly_variable_genes_kws={}):
    """
    Parameters
    ----------
    palette : None or collection
        Set to None when running for the second time to force usage of previous palette
    """
    # Get highly variable genes
    _highly_variable_genes.highly_variable_genes(adata, flavor='cell_ranger', **highly_variable_genes_kws)
    print('\n','Number of highly variable genes: {:d}'.format(np.sum(adata.var['highly_variable'])))
    
    # Compute new PCA and UMAP
    sc.tl.pca(adata, svd_solver='arpack')
#     sc.pp.neighbors(adata)
    sc.external.pp.bbknn(adata, batch_key='individual')
    sc.tl.umap(adata)
    
    # Plot new PCA
    
    sc.pl.umap(adata, color='tissue', palette=palette)
    sc.pl.umap(adata, color='individual')
    return adata


def clean_free_annotation(adata):
        # Clean free annotation
    adata.obs['free_annotation_cleaned'] = adata.obs.free_annotation.str.split(" \(").str[0]
    adata.obs['free_annotation_cleaned'] = adata.obs.free_annotation_cleaned.map(
        lambda x: 'doublet' if x.startswith('doublet') else x)
    adata.obs['free_annotation_cleaned'] = adata.obs.free_annotation_cleaned.map(
        lambda x: 'NA' if x.startswith('unknown') else x)
    return adata


def differential_expression(adata, celltype, col='cell_ontology_class'):
    # --- Differential expression --- #
    # Add categorial column for differential expression and plotting
    is_celltype = adata.obs[col] == celltype
    print(f"is_celltype.sum(): {is_celltype.sum()}")

    is_celltype_col = f'is_{celltype.lower().replace(" ", "_")}'
    is_celltype_category_col = f'is_{celltype.lower().replace(" ", "_")}_category'
    adata.obs[is_celltype_col] = is_celltype
    adata.obs[is_celltype_category_col] = adata.obs[is_celltype_col].map(
        lambda x: celltype if x else 'other cell')
    
    plot_umap_ncounts_ngenes(adata, is_celltype_col)
    
    adata, diff_genes = rank_genes_groups_and_plot(adata, is_celltype_category_col)
    return adata, diff_genes
#     # Grep for the gene name in the ncbi GTF to see if there is anything useful written in the notes
#     for col_name, col_series in diff_genes.head(10).iteritems():
#         print(f'\n--- {col_name} ---')
#         for gene_name in col_series.values:
#             ! grep $gene_name $ncbi_gtf | grep -m 1 product
            

def rank_genes_groups_and_plot(adata, groupby):
    sc.tl.rank_genes_groups(adata, groupby=groupby)
    # Plot the ranked genes per group
    sc.pl.rank_genes_groups(adata)
    
    sc.pl.rank_genes_groups_stacked_violin(adata)
    
    try:
        sc.pl.rank_genes_groups_violin(adata)
    except:
        # Sometimes this plot doesn't work and I don't know why
        pass
    
    diff_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    return adata, diff_genes
            
            
def plot_umap_ncounts_ngenes(adata, groupby):
    sc.pl.umap(adata, color=groupby)

    sc.pl.umap(adata, color='n_counts')
    sc.pl.violin(adata, 'n_counts', groupby=groupby)
    sc.pl.umap(adata, color='n_genes')
    sc.pl.violin(adata, 'n_genes', groupby=groupby)


def filter_cells(
    data: AnnData,
    min_counts: Optional[int] = None,
    min_genes:  Optional[int] = None,
    max_counts: Optional[int] = None,
    max_genes:  Optional[int] = None,
    inplace: bool = True,
    copy: bool = False,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """\
    Filter cell outliers based on counts and numbers of genes expressed.
    For instance, only keep cells with at least `min_counts` counts or
    `min_genes` genes expressed. This is to filter measurement outliers,
    i.e. “unreliable” observations.
    Only provide one of the optional parameters `min_counts`, `min_genes`,
    `max_counts`, `max_genes` per call.
    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` × `n_vars`.
        Rows correspond to cells and columns to genes.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_genes
        Minimum number of genes expressed required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_genes
        Maximum number of genes expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.
    Returns
    -------
    Depending on `inplace`, returns the following arrays or directly subsets
    and annotates the data matrix:
    cells_subset
        Boolean index mask that does filtering. `True` means that the
        cell is kept. `False` means the cell is removed.
    number_per_cell
        Depending on what was tresholded (`counts` or `genes`),
        the array stores `n_counts` or `n_cells` per gene.
    Examples
    --------
    >>> import scanpy as sc
    >>> adata = sc.datasets.krumsiek11()
    >>> adata.n_obs
    640
    >>> adata.var_names
    ['Gata2' 'Gata1' 'Fog1' 'EKLF' 'Fli1' 'SCL' 'Cebpa'
     'Pu.1' 'cJun' 'EgrNab' 'Gfi1']
    >>> # add some true zeros
    >>> adata.X[adata.X < 0.3] = 0
    >>> # simply compute the number of genes per cell
    >>> sc.pp.filter_cells(adata, min_genes=0)
    >>> adata.n_obs
    640
    >>> adata.obs['n_genes'].min()
    1
    >>> # filter manually
    >>> adata_copy = adata[adata.obs['n_genes'] >= 3]
    >>> adata_copy.obs['n_genes'].min()
    >>> adata.n_obs
    554
    >>> adata.obs['n_genes'].min()
    3
    >>> # actually do some filtering
    >>> sc.pp.filter_cells(adata, min_genes=3)
    >>> adata.n_obs
    554
    >>> adata.obs['n_genes'].min()
    3
    """
    if copy:
       logg.warning('`copy` is deprecated, use `inplace` instead.')
    n_given_options = sum(
        option is not None for option in
        [min_genes, min_counts, max_genes, max_counts])
    if n_given_options != 1:
        raise ValueError(
            'Only provide one of the optional parameters `min_counts`, '
            '`min_genes`, `max_counts`, `max_genes` per call.')
    if isinstance(data, AnnData):
        adata = data.copy() if copy else data
        cell_subset, number = materialize_as_ndarray(filter_cells(adata.X, min_counts, min_genes, max_counts, max_genes))
        if not inplace:
            return cell_subset, number
        if min_genes is None and max_genes is None: adata.obs['n_counts'] = number
        else: adata.obs['n_genes'] = number
        adata._inplace_subset_obs(cell_subset)
        return adata if copy else None
    X = data  # proceed with processing the data matrix
    min_number = min_counts if min_genes is None else min_genes
    max_number = max_counts if max_genes is None else max_genes
    number_per_cell = np.sum(X if min_genes is None and max_genes is None
                             else X > 0, axis=1)
    if issparse(X): number_per_cell = number_per_cell.A1
    if min_number is not None:
        cell_subset = number_per_cell >= min_number
    if max_number is not None:
        cell_subset = number_per_cell <= max_number

    s = np.sum(~cell_subset)
    if s > 0:
        msg = f'filtered out {s} cells that have '
        if min_genes is not None or min_counts is not None:
            msg += 'less than '
            msg += f'{min_genes} genes expressed' if min_counts is None else f'{min_counts} counts'
        if max_genes is not None or max_counts is not None:
            msg += 'more than '
            msg += f'{max_genes} genes expressed' if max_counts is None else f'{max_counts} counts'
        logg.info(msg)
    return cell_subset, number_per_cell


pancreatic_acinar = ['CPA1', 'CPA2', "PRSS2", 'CTRB1', "GP2"]
pancreatic_ductal = ['KRT19', "KRT18", "MUC1", "SPP1", "BICC1", "TFF3", "GPX2"]
pancreatic_alpha = ["GCG", "GC", "IRX1", "IRX2"]
pancreatic_beta = ['INS', 'MAFA', "PAX6", "PCSK1", "NEUROD1"]
pancreatic_combined = pancreatic_acinar + pancreatic_ductal + pancreatic_alpha + pancreatic_beta
pancreatic_v2 = ['CPA1', 'CPA2', "PRSS2", 'CTRB1', "GCG", "INS"]

lung_club = ["GPX2", "SCGB3A2", "TFF3"]
lung_club_v2 = ["SCGB3A2", "SCGB1A1","SCGB3A1","TFF3", "GPX2", "RETNLB", "CYP2F2", "CCKAR", "CTSE", "SFTA1P", "MGP", "CAV1"]
lung_basal_v2 = ["S100A2","DAPL1","ISLR","DLK2", "LOC105872035","KRT5", "TP63", "NGFR",  "ADH7", "SNCA","LOC105862085", "KRT17"]
lung_basal = ['GPX2', "S100A2", "DAPL1", "TFF3"]
#     note pancreatic ductal cells are TFF3+, GPX2+low

celltype_markers = dict(pancreatic_acinar=pancreatic_acinar,
                       pancreatic_ductal=pancreatic_ductal,
                       pancreatic_alpha=pancreatic_alpha,
                       pancreatic_beta=pancreatic_beta,
                        pancreatic_combined=pancreatic_combined,
                        pancreatic_v2=pancreatic_v2,
                       lung_club=lung_club,
                        lung_club_v2=lung_club_v2,
                       lung_basal=lung_basal,
                       lung_basal_v2=lung_basal_v2)
# Filter for only markers present in the data

def get_present_celltype_markers(adata):
    celltype_markers = {k: sorted(adata.var.index.intersection(v)) for k, v in celltype_markers.items()}
    return celltype_markers
 

    
def plot_celltype_umaps(adata):
    celltype_markers = get_present_celltype_markers(adata)
    for key, genes in celltype_markers.items():
        sc.pl.umap(adata, color=genes, cmap='magma_r', frameon=False)

def plot_celltype_stacked_violins(adata):
    celltype_markers = get_present_celltype_markers(adata)
    for key, genes in celltype_markers.items():
        sc.pl.stacked_violin(adata, genes, groupby='tissue')

        
def compute_celltype_scores(adata, celltype_markers=celltype_markers, additional_groupbys=['animal', 'sequencing_run']):
#     use_raws = (True,) #False

    for key, genes in celltype_markers.items():
#         for use_raw in use_raws:
#             suffix = 'raw' if use_raw else "normalized"
        score_name = f'{key}_score'
        sc.tl.score_genes(adata, genes, score_name=score_name, random_state=0)
        sc.pl.umap(adata, color=score_name, cmap='magma_r', return_fig=True)
#         fig, ax = plt.subplots(figsize=(16, 4))
        sc.pl.violin(adata, score_name, groupby='tissue', rotation=90)
#         fig.suptitle(key)
        for groupby in additional_groupbys:
            sc.pl.violin(adata, score_name, groupby=groupby, rotation=90)

            
            
            
def plot_non_tissue_scores(adata, tissue, score, groupbys=['tissue', 'animal']):
    adata_non_tissue = adata[adata.obs.tissue != tissue]
    
    # Do violinplot
    for groupby in groupbys:
        sc.pl.violin(adata_non_tissue, score, groupby=groupby, rotation=90)
    
    # Do barplot of number of cells
    for groupby in groupbys:
        df = adata_non_tissue.obs.loc[adata_non_tissue.obs[score] > 0]
        fig, ax = plt.subplots(figsize=(4.5, 3))
        palette = adata.uns[f'{groupby}_colors']
        sns.countplot(x=groupby, data=df, palette=palette)
        sns.despine()
        ax.set(ylabel='n_barcodes')
        if groupby == 'tissue':
            ax.tick_params(axis='x', labelrotation=90)


### Gene ontology enrichment
def get_diff_genes(adata):
    diff_genes = pd.DataFrame(adata.uns['rank_genes_groups']['names'])
    pvals = pd.DataFrame(adata.uns['rank_genes_groups']['pvals_adj'])
    #     diff_genes = diff_genes[pvals < diff_pval_thresh]
    return diff_genes


def do_go_enrichment_on_diff_other(adata):
    diff_genes = get_diff_genes(adata)
    go_enrichment = sc.queries.enrich(diff_genes['other cell'], org='mmurinus')
    return go_enrichment
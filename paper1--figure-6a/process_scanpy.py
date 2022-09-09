import anndata
import numpy as np
import pandas as pd
import scanpy as sc


def calculate_quality_control_covariates(adata):
    adata.obs['n_counts'] = adata.X.sum(1)
    adata.obs['log_counts'] = np.log(adata.obs['n_counts'])
    adata.obs['sqrt_counts'] = np.sqrt(adata.obs['n_counts'])
    adata.obs['n_genes'] = (adata.X > 0).sum(1)

    mt_gene_mask = np.flatnonzero([gene.startswith('MT-') for gene in adata.var_names])
    # the `.A1` is only necessary as X is sparse (to transform to a dense array after summing)
    adata.obs['mt_frac'] = np.sum(adata[:, mt_gene_mask].X, axis=1).A1/adata.obs['n_counts']
    return adata
    
    
    
#Data quality summary plots

def plot_quality_control_covariates(adata):
    p1 = sc.pl.scatter(adata, 'n_counts', 'n_genes', color='mt_frac', size=40)
    p2 = sc.pl.scatter(adata[adata.obs['n_counts']<15000], 'n_counts', 'n_genes', 
                       color='mt_frac', size=40)

def plot_sample_quality_violins(adata):
    #Sample quality plots
    rcParams['figure.figsize']=(7,7)
    t1 = sc.pl.violin(adata, 'n_counts',
                      #groupby='sample',
                      size=2, log=True, cut=0)
    t2 = sc.pl.violin(adata, 'mt_frac')
    
#Thresholding decision: counts
def plot_counts_thresholding(adata):

    rcParams['figure.figsize']=(20,5)
    fig_ind=np.arange(131, 134)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.6)

    p3 = sb.distplot(adata.obs['n_counts'], 
                     kde=False, 
                     ax=fig.add_subplot(fig_ind[0]))
    p4 = sb.distplot(adata.obs['n_counts'][adata.obs['n_counts']<4000], 
                     kde=False, bins=60, 
                     ax=fig.add_subplot(fig_ind[1]))
    p5 = sb.distplot(adata.obs['n_counts'][adata.obs['n_counts']>10000], 
                     kde=False, bins=60, 
                     ax=fig.add_subplot(fig_ind[2]))
    plt.show()

    
    
#Thresholding decision: genes
def plot_gene_thresholding(adata):
    rcParams['figure.figsize']=(20,5)
    fig_ind=np.arange(131, 133)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.6) #create a grid for subplots

    p6 = sb.distplot(adata.obs['n_genes'], kde=False, bins=60, ax=fig.add_subplot(fig_ind[0]))

    p7 = sb.distplot(adata.obs['n_genes'][adata.obs['n_genes']<1000], 
                     kde=False, bins=60, ax=fig.add_subplot(fig_ind[1]))
    plt.show()

    
def filter_cells(adata:anndata.AnnData, min_counts:int=1000, 
                 max_counts:int=0, 
                 min_genes:int=100, max_genes:int=0):
    # Filter cells according to identified QC thresholds:
    print('Total number of cells: {:d}'.format(adata.n_obs))

    sc.pp.filter_cells(adata, min_counts = min_counts)
    print('Number of cells after min count filter: {:d}'.format(adata.n_obs))

    if max_counts > 0:
        sc.pp.filter_cells(adata, max_counts = max_counts)
        print('Number of cells after max count filter: {:d}'.format(adata.n_obs))

    sc.pp.filter_cells(adata, min_genes = min_counts)
    print('Number of cells after gene filter: {:d}'.format(adata.n_obs))
    return adata


def filter_genes(adata: anndata.AnnData):

    print('Total number of genes: {:d}'.format(adata.n_vars))

    # Min 20 cells - filters out 0 count genes
    sc.pp.filter_genes(adata, min_cells=2)
    print('Number of genes after cell filter: {:d}'.format(adata.n_vars))
    return adata


def normalize_counts(adata, method='log'):
    if method == 'log':
        sc.pp.log1p(adata)
    elif method == 'sqrt':
        sc.pp.sqrt(adata)
    else:
        print(f"Method '{method}' is not a valid normalization method")

def get_and_plot_highly_variable_genes(adata, n_top_genes=2000, flavor='cell_ranger'):
    sc.pp.highly_variable_genes(adata, flavor=flavor, n_top_genes=n_top_genes)

    print('\n','Number of highly variable genes: {:d}'.format(
        np.sum(adata.var['highly_variable'])))

    rcParams['figure.figsize']=(10,5)
    sc.pl.highly_variable_genes(adata)
    
    
def compute_dimensionality_reduction(adata, n_comps=50, use_highly_variable=True, 
                                     svd_solver='arpack'):

    sc.pp.pca(adata, n_comps=n_comps, use_highly_variable=True, svd_solver='arpack')
    sc.pp.neighbors(adata)

    sc.tl.tsne(adata) #Note n_jobs works for MulticoreTSNE, but not regular implementation)
    sc.tl.umap(adata)
    sc.tl.diffmap(adata)
    sc.tl.draw_graph(adata)
    
    
def plot_dimensionality_reduction(adata):
    rcParams['figure.figsize']=(20,10)
    fig_ind=np.arange(231, 237)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.6)

    p10 = sc.pl.pca_scatter(adata, color='n_counts', ax=fig.add_subplot(fig_ind[0]), show=False)
    p11 = sc.pl.tsne(adata, color='n_counts', ax=fig.add_subplot(fig_ind[1]), show=False)
    p12 = sc.pl.umap(adata, color='n_counts', ax=fig.add_subplot(fig_ind[2]), show=False)
    p13 = sc.pl.diffmap(adata, color='n_counts', components=['1,2'], ax=fig.add_subplot(fig_ind[3]),show=False)
    p14 = sc.pl.diffmap(adata, color='n_counts', components=['1,3'], ax=fig.add_subplot(fig_ind[4]), show=False)
    p15 = sc.pl.draw_graph(adata, color='n_counts', ax=fig.add_subplot(fig_ind[5]), show=False)

    plt.show()
    
    
def score_cell_cycle(adata, cell_cycle_genes=None):
    if cell_cycle_genes is None:
        cell_cycle_txt = 'https://raw.githubusercontent.com/theislab/scanpy-demo-czbiohub/master/Macosko_cell_cycle_genes.txt'
        cell_cycle_genes = pd.read_csv(cell_cycle_txt, sep='\t')

    s_genes = cell_cycle_genes['S']
    g2m_genes = cell_cycle_genes['G2.M']
    
    # May have provided a dictionary of lists, which don't have dropna capabilities
    try:
        s_genes = s_genes.dropna()
        g2m_genes = g2m_genes.dropna()
    except AttributeError:
        pass

    s_genes_hvg = adata.var_names[np.in1d(adata.var_names, s_genes)]
    g2m_genes_hvg = adata.var_names[np.in1d(adata.var_names, g2m_genes)]

    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes_hvg, g2m_genes=g2m_genes_hvg)
    print(len(s_genes_hvg))
    print(len(g2m_genes_hvg))
    adata.obs['phase'].value_counts()

    return adata


def plot_cell_cycle(adata):
    rcParams['figure.figsize']=(5,5)
    sc.pl.umap(adata, color=['S_score'], use_raw=False)
    sc.pl.umap(adata, color=['G2M_score'], use_raw=False)
    sc.pl.umap(adata, color=['phase'], use_raw=False)
    sc.pl.umap(adata, color='MKI67')

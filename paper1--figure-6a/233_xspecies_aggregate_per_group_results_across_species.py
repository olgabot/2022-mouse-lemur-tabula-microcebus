#!/usr/bin/env python
# coding: utf-8

# In[1]:


from collections import defaultdict
import glob
import os

import anndata
import pandas as pd
import numpy as np
import scanpy
import scanpy as sc

import xspecies

scanpy.settings.verbosity = 3
#sc.set_figure_params(dpi=200, dpi_save=300)
scanpy.logging.print_versions()
scanpy.set_figure_params(frameon=False, color_map='magma_r')


# # Define outdirs

# In[2]:


outdir_gdrive = '/home/olga/googledrive/TabulaMicrocebus/data/cross-species/within-species-de'
outdir_local = '/home/olga/data_sm/tabula-microcebus/data-objects/cross-species/within-species-de'
get_ipython().system(' mkdir $outdir_gdrive $outdir_local')
outdirs = outdir_local, outdir_gdrive
get_ipython().system('ls -lha $outdir_local')


# # Iterate over h5ads

# In[3]:


adatas = defaultdict(dict)


for h5ad in glob.glob(f'{outdir_local}/*diffexpr*defaults.h5ad'):
    basename = os.path.basename(h5ad)
    split = basename.split('__')
    species = split[0].split('_')[0]
    group = split[1].split('_')[1]
    print(basename)
    print(f"group: {group}")
    print(f"species: {species}")
        
    ad = sc.read(h5ad, cache=True)
    adatas[group][species] = ad


# ## Create per-group species aggregated

# In[11]:


import xspecies

class XspeciesAdatas:
    def __init__(self, group, human=None, mouse=None, lemur=None):
        self.group = group
        self.human = human
        self.mouse = mouse
        self.lemur = lemur
        
        self.adatas = {'human': self.human, 'mouse': self.mouse, 'lemur': self.lemur}
        
        self.de_results = self.aggregate_de_results()

        
    @staticmethod 
    def _single_adata_get_de_single_result(adata, result):
        if result == 'names':
            name = 'gene_name'
            extractor = xspecies.CrossSpeciesComparison.extract_de_names
        elif result == 'scores':
            name = 'de_score'
            extractor = xspecies.CrossSpeciesComparison.extract_de_scores
        elif result == 'logfoldchanges':
            name = 'de_logfoldchange'
            extractor = xspecies.CrossSpeciesComparison.extract_de_logfoldchanges
        elif result == 'pvals':
            name = 'de_pval_adj'
            extractor = xspecies.CrossSpeciesComparison.extract_de_pvals_adj
        else:
            raise ValueError(f"{result} is not a valid result name")
        results = extractor(adata).unstack()
        results.name = name
        results = results.to_frame()
        return results
    
    def _single_adata_get_de_results(self, adata, species, result_names=('names', 'scores', 'pvals', 'logfoldchanges')):

        dfs = []

        for name in result_names:
            df = self._single_adata_get_de_single_result(adata, name)
            dfs.append(df)
            
        # Column bind, because the row names (index) are shared
        de_results = pd.concat(dfs, axis=1)
        de_results = de_results.reset_index()
        de_results = de_results.rename(columns={'level_0': group, 'level_1': 'de_rank'})
        de_results['species'] = species
#         de_results.head()
        return de_results

    def aggregate_de_results(self):
        dfs = []
        
        for species, adata in self.adatas.items():
            if adata is None:
                print(f'{species} has no adata - skipping')
                continue
            df = self._single_adata_get_de_results(adata, species)
            dfs.append(df)
            
        # Row bind because columns are shared
        de_results = pd.concat(dfs)
        return de_results

        
group = 'compartment'
compartment = XspeciesAdatas(group, **adatas[group])
compartment.de_results.head()


# In[23]:


adatas['broad']


# In[24]:


df = compartment.de_results.query('compartment == "endothelial"')
df.head()


# In[25]:


df.pivot_table(columns='species', index='de_rank', values=['gene_name'], observed=False, aggfunc=lambda x: x)


# In[26]:


def pivot_by_species(df):
    pivoted = df.pivot_table(columns='species', index='de_rank', 
                             values=['gene_name', 'de_score', 'de_pval_adj', 'de_logfoldchange'], 
                             aggfunc=lambda x: x)
    return pivoted


# ## Write aggregated results to excel

# In[27]:


with pd.ExcelWriter(f"{outdir_gdrive}/across_species_{compartment.group}.xlsx") as writer:
    for name, df in compartment.de_results.groupby(compartment.group):
        pivoted = pivot_by_species(df)
        pivoted.to_excel(writer, sheet_name=name)


# # Iterate over all compartments and make all fiels

# In[28]:


for group, ads in adatas.items():
    print(f'-- {group} --')
    print(ads)
    group_xspecies = XspeciesAdatas(group, **ads)
    xlsx = f"{outdir_gdrive}/across_species_{group_xspecies.group}.xlsx"
    with pd.ExcelWriter(xlsx) as writer:
        for name, df in group_xspecies.de_results.groupby(group_xspecies.group):
            name_cleaned = name.replace('/', '-slash-')
            pivoted = pivot_by_species(df)
            pivoted.to_excel(writer, sheet_name=name_cleaned)
    print(f"\tWrote {xlsx}")


#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')

get_ipython().run_line_magic('autoreload', '2')

from io import StringIO

import anndata
import pandas as pd
import numpy as np
import scanpy
import scanpy as sc

scanpy.settings.verbosity = 3
#sc.set_figure_params(dpi=200, dpi_save=300)
scanpy.logging.print_versions()
scanpy.set_figure_params(frameon=False, color_map='magma_r')

outdir = '/home/olga/googledrive/TabulaMicrocebus/data/cross-species'
get_ipython().system(' ls -lha $outdir')


# # Define outdirs

# In[2]:


outdir_gdrive = '/home/olga/googledrive/TabulaMicrocebus/data/cross-species/within-species-de'
outdir_local = '/home/olga/data_sm/tabula-microcebus/data-objects/cross-species/within-species-de'
get_ipython().system(' mkdir $outdir_gdrive $outdir_local')
outdirs = outdir_local, outdir_gdrive
get_ipython().system('ls -lha $outdir_local')


# # Load into Scanpy

# # Mouse Lemur raw counts
# 
# Created with this pull request: https://github.com/czbiohub/tabula-microcebus/pull/15

# In[3]:


lemur_folder = '/home/olga/data_sm/tabula-microcebus/data-objects/10x'
get_ipython().system(' ls -lha $lemur_folder')


# ## Read all raw count data

# In[4]:


get_ipython().run_cell_magic('time', '', "\nh5ad = f'{lemur_folder}/tabula-microcebus--10x--counts--min-51-genes--min-101-counts--trnas-summed.h5ad'\nlemur = scanpy.read(h5ad, cache=True)\nlemur")


# ## Subset to lung

# In[5]:


lemur_lung = lemur[lemur.obs.tissue == "Lung"]
lemur_lung


# ### Look at lemur lung.obs.head()

# In[6]:


lemur_lung.obs.head()


# ## Read all species object with unified compartments

# In[7]:


get_ipython().run_cell_magic('time', '', "h5ad = f'{outdir}/concatenated__human-lung--lemur-lung--mouse-lung__10x__one2one_orthologs__unified_compartments.h5ad'\nadata = scanpy.read_h5ad(h5ad)\nadata")


# ### Look at adata.obs.head()

# In[8]:


adata.obs.head()


# ### Remove "-lemur" from combined annotations

# In[9]:


lemur_obs_from_concatenated = adata.obs.query('species_batch == "lemur"')
lemur_obs_from_concatenated.index = lemur_obs_from_concatenated.index.str.split('-lemur').str[0]
lemur_obs_from_concatenated.head()


# ## Add unified compartments

# In[10]:


unified_cols = ['narrow_group', 'broad_group', 'compartment_group']


# In[19]:


obs_with_compartments = lemur_lung.obs.join(lemur_obs_from_concatenated[unified_cols].astype(str))
obs_with_compartments.head()


# In[20]:


obs_with_compartments[unified_cols].notnull().sum()


# In[27]:


obs_with_compartments['narrow_group'].value_counts()


# ## Make `lemur_lung_new_obs` with compartment groups and cleaned up narrow groups

# In[30]:


lemur_lung_new_obs = lemur_lung.copy()
lemur_lung_new_obs.obs = obs_with_compartments
lemur_lung_new_obs = lemur_lung_new_obs[lemur_lung_new_obs.obs.groupby('narrow_group').filter(lambda x: len(x) >= 3).index]
lemur_lung_new_obs = lemur_lung_new_obs[lemur_lung_new_obs.obs.query('narrow_group != "nan"').index]
lemur_lung_new_obs


# ## Write newly created object to file

# In[35]:


for d in outdirs:
    h5ad = f'{d}/lemur_lung_from_tabula-microcebus__unified_compartments.h5ad'
    print(h5ad)
    lemur_lung_new_obs.write(h5ad)


# # do differential expression
# 

# In[36]:


import diffexpr


# In[37]:


prefix = 'lemur_lung'


# ## narrow group

# In[38]:


get_ipython().run_cell_magic('time', '', '\ngroup = \'narrow_group\'\nsc.tl.rank_genes_groups(lemur_lung_new_obs, group)\n\nfor d in outdirs:\n    h5ad = f"{d}/{prefix}__diffexpr_{group}__defaults.h5ad"\n    lemur_lung_new_obs.write(h5ad)\n\n# mouse_lung_new_obs.uns[\'rank_genes_groups\']\n\n# sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)')


# ## broad group

# In[39]:


get_ipython().run_cell_magic('time', '', '\ngroup = \'broad_group\'\nsc.tl.rank_genes_groups(lemur_lung_new_obs, group)\n\nfor d in outdirs:\n    h5ad = f"{d}/{prefix}__diffexpr_{group}__defaults.h5ad"\n    lemur_lung_new_obs.write(h5ad)\n\n# mouse_lung_new_obs.uns[\'rank_genes_groups\']\n\n# sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)')


# ## compartment group

# In[40]:


get_ipython().run_cell_magic('time', '', '\ngroup = \'compartment_group\'\nsc.tl.rank_genes_groups(lemur_lung_new_obs, group)\n\nfor d in outdirs:\n    h5ad = f"{d}/{prefix}__diffexpr_{group}__defaults.h5ad"\n    lemur_lung_new_obs.write(h5ad)\n\n# mouse_lung_new_obs.uns[\'rank_genes_groups\']\n\n# sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)')


# # Logistic regression

# ## compartment group

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ngroup = 'compartment_group'\nsc.tl.rank_genes_groups(lemur_lung_new_obs, group)\n# diffexpr.do_differential_expression(mouse_lung_new_obs, group, n_jobs=16)")


# In[ ]:





# In[16]:


group = 'compartment_group'

diffexpr.do_differential_expression(lemur_lung_new_obs, group)


# ### Write results to file

# In[17]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}.h5ad"
    lemur_lung_new_obs.write(h5ad)
#     mouse_lung_new_obs.write_csvs(f'{prefix}__diffexpr__{group}')


# ## Broad group

# In[ ]:


group = 'broad_group'

diffexpr.do_differential_expression(lemur_lung_new_obs, group, n_jobs=16)


# In[ ]:


lemur_lung_new_obs.uns['rank_genes_groups']


# ### Write results to file

# In[ ]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}.h5ad"
    lemur_lung_new_obs.write(h5ad)
#     lemur_lung_new_obs.write_csvs(f'{prefix}__diffexpr__{group}')


# ## narrow group

# In[ ]:


group = 'narrow_group'
diffexpr.do_differential_expression(lemur_lung_new_obs, group) 


# ### Write results to file

# In[ ]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}.h5ad"
    lemur_lung_new_obs.write(h5ad)
#     lemur_lung_new_obs.write_csvs(f'{prefix}__diffexpr__{group}')


# In[ ]:





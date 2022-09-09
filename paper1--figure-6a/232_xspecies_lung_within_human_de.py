#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

# ## Read preprocessed Human h5ad

# In[3]:


human_folder = '/home/olga/googledrive/TabulaMicrocebus/data/human-lung-cell-atlas--from-kyle'
get_ipython().system(' ls -lha $human_folder')


# In[4]:


get_ipython().run_cell_magic('time', '', '\nh5ad = f"{human_folder}/droplet_normal_lung_blood_P1-3.h5ad"\nhuman = scanpy.read(h5ad, cache=True)\nhuman')


# ## Subset to lung

# In[5]:


human_lung = human[human.obs.tissue == "lung"]
human_lung


# ### Look at lemur lung.obs.head()

# In[6]:


human_lung.obs.head()


# ## Read all species object with unified compartments

# In[16]:


get_ipython().run_cell_magic('time', '', "h5ad = f'{outdir}/concatenated__human-lung--lemur-lung--mouse-lung__10x__one2one_orthologs__unified_compartments.h5ad'\nadata = scanpy.read_h5ad(h5ad)\nadata")


# ### Look at adata.obs.head()

# In[17]:


adata.obs.head()


# ### Remove "-human" from combined annotations

# In[18]:


obs_from_concatenated = adata.obs.query('species_batch == "human"')
obs_from_concatenated.index = obs_from_concatenated.index.str.split('-human').str[0]
obs_from_concatenated.head()


# ## Add unified compartments

# In[19]:


unified_cols = ['narrow_group', 'broad_group', 'compartment_group']


# In[20]:


obs_with_compartments = human_lung.obs.join(obs_from_concatenated[unified_cols].astype(str))
obs_with_compartments.head()


# In[21]:


obs_with_compartments[unified_cols].notnull().sum()


# In[22]:


obs_with_compartments['compartment_group'].value_counts()


# In[23]:


obs_with_compartments['narrow_group'].value_counts()


# In[25]:


human_lung_new_obs = human_lung.copy()
human_lung_new_obs.obs = obs_with_compartments
human_lung_new_obs = human_lung_new_obs[human_lung_new_obs.obs.groupby('narrow_group').filter(lambda x: len(x) >= 3).index]
human_lung_new_obs = human_lung_new_obs[human_lung_new_obs.obs.query('narrow_group != "nan"').index]


# ## Write newly created object to file

# In[26]:


for d in outdirs:
    h5ad = f'{d}/human_lung_from_hlca__unified_compartments.h5ad'
    print(h5ad)
    human_lung_new_obs.write(h5ad)


# # do differential expression
# 

# In[27]:


import diffexpr


# In[28]:


# human_lung_new_obs.uns['rank_genes_groups']


# In[29]:


prefix = 'human_lung'


# ## narrow group

# In[30]:


get_ipython().run_cell_magic('time', '', '\ngroup = \'narrow_group\'\nsc.tl.rank_genes_groups(human_lung_new_obs, group)\n\nfor d in outdirs:\n    h5ad = f"{d}/{prefix}__diffexpr_{group}__defaults.h5ad"\n    human_lung_new_obs.write(h5ad)\n\n# mouse_lung_new_obs.uns[\'rank_genes_groups\']\n\n# sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)')


# ## broad group

# In[31]:


get_ipython().run_cell_magic('time', '', '\ngroup = \'broad_group\'\nsc.tl.rank_genes_groups(human_lung_new_obs, group)\n\nfor d in outdirs:\n    h5ad = f"{d}/{prefix}__diffexpr_{group}__defaults.h5ad"\n    human_lung_new_obs.write(h5ad)\n\n# mouse_lung_new_obs.uns[\'rank_genes_groups\']\n\n# sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)')


# ## compartment group

# In[32]:


get_ipython().run_cell_magic('time', '', '\ngroup = \'compartment_group\'\nsc.tl.rank_genes_groups(human_lung_new_obs, group)\n\nfor d in outdirs:\n    h5ad = f"{d}/{prefix}__diffexpr_{group}__defaults.h5ad"\n    human_lung_new_obs.write(h5ad)\n\n# mouse_lung_new_obs.uns[\'rank_genes_groups\']\n\n# sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)')


# # Logistic regression

# ## compartment group

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ngroup = 'compartment_group'\nsc.tl.rank_genes_groups(human_lung_new_obs, group)\n\nsc.pl.rank_genes_groups(human_lung_new_obs, sharey=False)")


# In[ ]:


group = 'compartment_group'

diffexpr.do_differential_expression(human_lung_new_obs, group, n_jobs=16)


# ### Write results to file

# In[ ]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}.h5ad"
    human_lung_new_obs.write(h5ad)
#     human_lung_new_obs.write_csvs(f'{prefix}__diffexpr__{group}')


# ## narrow group

# In[ ]:





# In[ ]:


group = 'narrow_group'
diffexpr.do_differential_expression(human_lung_new_obs, group) 


# ### Write results to file

# In[ ]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}.h5ad"
    human_lung_new_obs.write(h5ad)
#     human_lung_new_obs.write_csvs(f'{prefix}__diffexpr__{group}')


# ## Broad group

# In[ ]:


group = 'broad_group'

do_differential_expression(human_lung_new_obs, group)


# In[ ]:


human_lung_new_obs.uns['rank_genes_groups']


# ### Write results to file

# In[ ]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}.h5ad"
    human_lung_new_obs.write(h5ad)
#     human_lung_new_obs.write_csvs(f'{prefix}__diffexpr__{group}')


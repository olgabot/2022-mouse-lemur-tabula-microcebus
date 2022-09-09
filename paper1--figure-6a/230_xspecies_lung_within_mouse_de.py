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

import diffexpr

scanpy.settings.verbosity = 3
#sc.set_figure_params(dpi=200, dpi_save=300)
scanpy.logging.print_versions()
scanpy.set_figure_params(frameon=False, color_map='magma_r')

outdir = '/home/olga/googledrive/TabulaMicrocebus/data/cross-species'
# ! ls -lha $outdir


# # Define outdirs

# In[2]:


outdir_gdrive = '/home/olga/googledrive/TabulaMicrocebus/data/cross-species/within-species-de'
outdir_local = '/home/olga/data_sm/tabula-microcebus/data-objects/cross-species/within-species-de'
get_ipython().system(' mkdir $outdir_gdrive $outdir_local')
outdirs = outdir_local, outdir_gdrive
get_ipython().system('ls -lha $outdir_local')


# # Load into Scanpy

# ## Read **original** Mouse Raw object with all tissues -- not filtered for orthologous genes or anything

# In[3]:


ll /home/olga/data_sm/czb-tabula-muris-senis/Data-objects


# In[4]:


get_ipython().run_cell_magic('time', '', '\nh5ad = \'/home/olga/data_sm/czb-tabula-muris-senis/Data-objects/tabula-muris-senis-droplet-official-raw-obj--no-duplicate-barcodes-per-seq-run.h5ad\'\nmouse = scanpy.read_h5ad(h5ad)\nprint(mouse)\nmouse_lung = mouse[mouse.obs.tissue == "Lung"]\nmouse_lung')


# ### Look at mouse.obs.head()

# In[5]:


mouse_lung.obs.head()


# ## Read all species object with unified compartments

# In[6]:


get_ipython().run_cell_magic('time', '', "h5ad = f'{outdir}/concatenated__human-lung--lemur-lung--mouse-lung__10x__one2one_orthologs__unified_compartments.h5ad'\nprint(h5ad)\nadata = scanpy.read_h5ad(h5ad)\nadata")


# ### Look at adata.obs.head()

# In[7]:


adata.obs.head()


# ### Remove "-mouse" from combined annotations

# In[8]:


mouse_obs_from_concatenated = adata.obs.query('species_batch == "mouse"')
mouse_obs_from_concatenated.index = mouse_obs_from_concatenated.index.str.split('-mouse').str[0]
mouse_obs_from_concatenated.head()


# ## Add unified compartments

# In[23]:


obs_with_compartments = mouse_lung.obs.join(mouse_obs_from_concatenated[['narrow_group', 'broad_group', 'compartment_group']].astype(str))
obs_with_compartments.head()


# In[24]:


obs_with_compartments.compartment_group.value_counts()


# In[33]:


obs_with_compartments.narrow_group.value_counts()


# ## Create new object

# In[25]:


mouse_lung_new_obs = mouse_lung.copy()
mouse_lung_new_obs.obs = obs_with_compartments
mouse_lung_new_obs


# ## Write newly created object to file

# In[26]:


for d in outdirs:
    h5ad = f'{d}/mouse_lung_from_tabula-muris-senis__unified_compartments.h5ad'
    print(h5ad)
    mouse_lung_new_obs.write(h5ad)


# In[ ]:


mouse_lung_new_obs_min_cells_per_group = mouse_lung_new_obs[mouse_lung_new_obs.groupby('')]


# # do differential expression
# 

# In[27]:


prefix = 'mouse_lung'


# ## compartment group

# In[28]:


mouse_lung_new_obs.obs.head()


# In[40]:


get_ipython().run_cell_magic('time', '', "\ngroup = 'compartment_group'\nsc.tl.rank_genes_groups(mouse_lung_new_obs, group)\n\n# sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)")


# ### Write results to file

# In[41]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}__defaults.h5ad"
    mouse_lung_new_obs.write(h5ad)
#     mouse_lung_new_obs.write_csvs(f'{prefix}__diffexpr__{group}')


# In[ ]:


sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)


# ## Narrow group

# In[42]:


get_ipython().run_cell_magic('time', '', "\ngroup = 'narrow_group'\nsc.tl.rank_genes_groups(mouse_lung_new_obs, group)")


# In[44]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}__defaults.h5ad"
    mouse_lung_new_obs.write(h5ad)

# mouse_lung_new_obs.uns['rank_genes_groups']

# sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)


# ## Broad group

# In[43]:


get_ipython().run_cell_magic('time', '', '\ngroup = \'broad_group\'\nsc.tl.rank_genes_groups(mouse_lung_new_obs, group)\n\nfor d in outdirs:\n    h5ad = f"{d}/{prefix}__diffexpr_{group}__defaults.h5ad"\n    mouse_lung_new_obs.write(h5ad)\n\nmouse_lung_new_obs.uns[\'rank_genes_groups\']\n\n# sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)')


# In[46]:


# mouse_lung_new_obs.uns['rank_genes_groups']


# In[47]:


group


# In[48]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}__defaults.h5ad"
    mouse_lung_new_obs.write(h5ad)

# mouse_lung_new_obs.uns['rank_genes_groups']

# sc.pl.rank_genes_groups(mouse_lung_new_obs, sharey=False)


# # Logistic regression method

# In[ ]:


mouse_lung_new_obs.uns['rank_genes_groups']


# ### Write results to file

# In[ ]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}.h5ad"
    mouse_lung_new_obs.write(h5ad)
#     mouse_lung_new_obs.write_csvs(f'{prefix}__diffexpr__{group}')


# ## Broad group

# In[ ]:


group = 'broad_group'

diffexpr.do_differential_expression(mouse_lung_new_obs, group, n_jobs=16)


# ### Write results to file

# In[ ]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}.h5ad"
    mouse_lung_new_obs.write(h5ad)
#     mouse_lung_new_obs.write_csvs(f'{prefix}__diffexpr__{group}')


# ## narrow group

# In[ ]:


get_ipython().run_cell_magic('time', '', "\ngroup = 'narrow_group'\ndo_differential_expression(mouse_lung_new_obs, group)")


# In[ ]:


sc.pl.rank_genes_groups(mouse_lung_new_obs)


# In[ ]:


mouse_lung_new_obs.uns['rank_genes_groups']


# ### Write results to file

# In[ ]:


for d in outdirs:
    h5ad = f"{d}/{prefix}__diffexpr_{group}.h5ad"
    mouse_lung_new_obs.write(h5ad)
#     mouse_lung_new_obs.write_csvs(f'{prefix}__diffexpr__{group}')


# In[ ]:





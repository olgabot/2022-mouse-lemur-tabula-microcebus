import itertools
import os
import random
from collections import defaultdict

import anndata
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Show HTML tables inline
from IPython.display import HTML, display
from joblib import Parallel, delayed
from scipy.stats import ttest_ind
from tqdm.auto import tqdm

import scanpy
import scanpy as sc

import plot_constants

# Set random seed since the correlation metric is stochastic
random.seed(2020)

idx = pd.IndexSlice

anndata.__version__

scanpy.settings.verbosity = 4

scanpy.logging.print_versions()
scanpy.set_figure_params(frameon=False, color_map="magma_r", transparent=True)

# ---- Set some constants ---
SPECIES_ORDER = ["Mouse", "Mouse lemur", "Human"]
SPECIES_BATCH_ORDER = ["mouse", "lemur", "human"]
N_SPECIES = len(SPECIES_ORDER)
SPECIES_PALETTE = sns.color_palette(n_colors=N_SPECIES)
sns.palplot(SPECIES_PALETTE)
ax = plt.gca()
ax.set(xticklabels=SPECIES_ORDER, xticks=np.arange(N_SPECIES))
SPECIES_BATCH_TO_COLOR_MAP = {
    'mouse': 'Blues', 'lemur': "YlOrBr", "human": "Greens"}
SPECIES_TO_COLOR_MAP = {
    'Mouse': 'Blues', 'Mouse lemur': "YlOrBr", "Human": "Greens"}


compartments = [
    'endothelial', 'epithelial', 'lymphoid', 'myeloid', 'stromal']
n_compartments = len(compartments)
compartment_colors = sns.color_palette("Set2", n_colors=n_compartments)
sns.palplot(compartment_colors)
compartment_to_color = dict(zip(compartments, compartment_colors))
ax = plt.gca()
ax.set(xticklabels=compartments, xticks=np.arange(n_compartments))


class CrossSpeciesComparison:
    def __init__(
        self,
        adata,
        groups_col,
        species_col="species",
        species=plot_constants.SPECIES_ORDER,
        ref_species="Mouse",
        subgroup_col=None,
        min_cells_per_species_per_group=20,
        species_to_color_map=SPECIES_TO_COLOR_MAP,
        technical_artifcat_genes_to_remove=None
    ):
        """compare primate-specific signal within groups of 'col'

        'col' must be a valid column in adata.obs

        """
        self.adata_original = adata
        self.groups_col = groups_col
        self.species_col = species_col
        self.species = species
        # Reference species for differential expression
        self.ref_species = ref_species
        self.subgroup_col = subgroup_col
        self.min_cells = min_cells_per_species_per_group
        self.species_to_color_map = species_to_color_map

        # Ensure that species always appear in correct order
        self.adata_original.obs[species_col] = pd.Categorical(
            adata.obs.species, categories=self.species, ordered=True
        )
        self.n_species = len(self.species)
        self.adata_shared_subset = self.subset_adata_shared_groups()

    def preprocess(self):
        # Subset data to shared groups

        self.adata_shared_subset = self.compute_plot_pca_umap(
            self.adata_shared_subset, batch_key=self.species_col
        )
        self.make_subset_adatas()

    def run(self):
        self.do_differential_expression()
        self.plot_de_results()

    def make_subset_adatas(self, do_pca_umap=True):
        self._make_groupby_adatas(do_pca_umap=do_pca_umap)
        self._make_species_adatas(do_pca_umap=do_pca_umap)

    def _make_groupby_adatas(self, do_pca_umap=True):
        self.mini_adatas = self.make_mini_adatas(
            groupby=self.groups_col,
            subgroup_col=self.subgroup_col,
            do_pca_umap=do_pca_umap,
        )

    def _make_species_adatas(self, do_pca_umap=True):
        self.species_adatas = self.make_mini_adatas(
            self.species_col, subgroup_col=self.groups_col, do_pca_umap=do_pca_umap
        )

    def add_species_to_group_names(self):
        """For stacked violins -- make cell types that are the same color in
        different species, the same color"""
        ontology_species_order = []

        for ontology, species in itertools.product(
            adata.obs.cell_ontology_class.cat.categories,
            adata.obs.species.cat.categories,
        ):
            ontology_species = f"{ontology} ({species})"
            ontology_species_order.append(ontology_species)
        ontology_to_color = pd.Series(
            ad.uns["cell_ontology_class_colors"],
            index=ad.obs.cell_ontology_class.cat.categories,
        )
        ontology_species_to_color = pd.Series(
            dict(
                zip(
                    ad.obs[ontology_species_col],
                    ontology_to_color[
                        ad.obs[ontology_species_col].str.split(" \\(").str[0]
                    ],
                )
            )
        )
        ontology_species_to_color = ontology_species_to_color.sort_index()

    def subset_adata_shared_groups(self, verbose=True):
        self.obs_shared = self.adata_original.obs.groupby(
            self.groups_col).filter(
                lambda x: (x[self.species_col].nunique() == self.n_species) & (
                    x[self.species_col].value_counts() > self.min_cells).all()
        )
        if verbose:
            print(
                self.obs_shared.groupby(
                    [self.groups_col, self.species_col]).size())

        return self.adata_original[self.obs_shared.index]

    @staticmethod
    def compute_plot_pca_umap(
        adata,
        palette=None,
        highly_variable_genes_kws=dict(flavor="cell_ranger"),
        n_comps=50,
        batch_key="species",
        subgroup_col=None,
    ):
        """
        Parameters
        ----------
        palette : None or collection
            Set to None when running for the second time to force usage
            of previous palette
        """
        # Get highly variable genes
        try:
            sc.pp.highly_variable_genes(adata, **highly_variable_genes_kws)
            print(
                "\n",
                "Number of highly variable genes: {:d}".format(
                    np.sum(adata.var["highly_variable"])
                ),
            )
        except ValueError:
            print(
                "Scanpy had an error when computing highly variable genes,"
                " using previous ones"
            )

        # Compute new PCA and UMAP
        sc.tl.pca(adata, svd_solver="arpack", n_comps=n_comps)
        sc.external.pp.bbknn(adata, batch_key=batch_key)
        sc.tl.umap(adata)

        # Plot new PCA
        if subgroup_col is not None:
            sc.pl.umap(adata, color=subgroup_col, palette=palette)
        sc.pl.umap(adata, color=batch_key)
        return adata

    def make_mini_adatas(
        self, groupby, subgroup_col=None, verbose=True, do_pca_umap=True
    ):
        """Subset the adata on the obs, making a mini-adata for each one

        :param groupby:
        :param subgroup_col:
        :param verbose:
        :param do_pca_umap: bool
            If False, then don't compute PCA, bbknn, and UMAP. Useful
            if you just want to make dotplots
        :return:
        """
        adatas = {}

        for group, df in self.obs_shared.groupby(groupby):
            # Don't use empty groups
            if len(df) == 0:
                continue
            if verbose:
                self.print_group(group)
            if verbose:
                print(f"number of cells: {len(df)}")
            ad = self.adata_shared_subset[df.index]
            if do_pca_umap:
                ad = self.compute_plot_pca_umap(ad, subgroup_col=subgroup_col)
            adatas[group] = ad
        return adatas

    @staticmethod
    def extract_de_scores(adata):
        de_scores = pd.DataFrame(adata.uns["rank_genes_groups"]["scores"])
        return de_scores

    @staticmethod
    def extract_de_names(adata):
        de_names = pd.DataFrame(adata.uns["rank_genes_groups"]["names"])
        return de_names

    @staticmethod
    def extract_de_logfoldchanges(adata):
        de_logfoldchanges = pd.DataFrame(
            adata.uns["rank_genes_groups"]["logfoldchanges"]
        )
        return de_logfoldchanges
    
    @staticmethod
    def extract_de_percent_expressing(adata):
        de_percent_expressing = pd.DataFrame(
            adata.uns["rank_genes_groups"]["pts"].copy()
        )
        return de_percent_expressing

    @staticmethod
    def extract_de_pvals_adj(adata):
        de_pvals_adj = pd.DataFrame(
            adata.uns["rank_genes_groups"]["pvals_adj"])
        return de_pvals_adj


    def remove_technical_artifact_gene_names(self, de_names):
        # RP[LS]\d+[A-Z]? FOS JUNB TMSB4X HBA1 HBB
        # Open the json file containing the list of strings that represent
        # the technical artifact genes to remove
        with open(self.technical_artifcat_genes_to_remove, "r") as f:
            genes_to_remove = json.load(f)
        regex_genes = []
        non_regex_genes = []
        # Extract genes containing ?[ etc as regex genes
        # and only alphabets as non-regex genes
        for gene in genes_to_remove:
            if string.isalpha():
                non_regex_genes.append(gene)
            else:
                regex_genes.append(gene)
        # Remove the rows in the dataframe containing human gene that matches the regex
        for regex_gene in regex_genes:
            de_names = de_names[~de_names['Human'].str.contains(
                regex_gene, flags=re.IGNORECASE, regex=True)]
        # Keep rows in the dataframe that contain human gene names that
        # do not match with the list of non regex genes
        de_names = de_names[~de_names.Human.isin(non_regex_genes)]
        return de_names

    def get_differentially_expressed_gene_names(
        self,
        ad,
        sort="score",
        abs_score_threshold=1e-3,
        primate_larger=True,
        select_genes=None,
        ignore_genes=None,
    ):
        """Get genes preferentially expressed in primate species (Human and Mouse Lemur)"""
        de_names = self.extract_de_names(ad)
        de_scores = self.extract_de_scores(ad)

        if primate_larger:
            score_mask = de_scores > abs_score_threshold
        else:
            score_mask = de_scores < -abs_score_threshold

        de_names = de_names[score_mask].dropna(how="all")
        de_scores = de_scores[score_mask].dropna(how="all")

        if sort == "alphabetically":
            shared_names = pd.Index(
                sorted(
                    de_names.Human[
                        de_names.Human.astype(str).isin(de_names["Mouse lemur"])
                    ].values
                )
            )
        elif sort == "score":
            # Make a sorting by differential expression score
            rows = de_names.Human.astype(str).isin(de_names["Mouse lemur"])
            median_scores = de_scores[["Human", "Mouse lemur"]].median(axis=1).loc[rows]

            shared_names = de_names.Human[rows].values
            median_scores = pd.Series(median_scores, index=shared_names)
            median_scores = median_scores.sort_values()
            shared_names = median_scores.index

        if ignore_genes is not None:
            shared_names = [x for x in shared_names if x not in ignore_genes]
        if select_genes is not None:
            # Subset to only genes in this group
            shared_names = [x for x in shared_names if x not in ignore_genes]

        return shared_names

    def species_dotplot(
            self, gene, groupby=None, standard_scale="var", **kwargs):
        if groupby is None:
            groupby = self.groupby

        for species, ad in self.species_adatas.items():
            print(f"--- {species} ---")
            sc.pl.dotplot(
                ad,
                gene,
                groupby=groupby,
                standard_scale=standard_scale,
                **kwargs,
            )

    def dotplot_multispecies(
        self,
        genes,
        max_genes=55,
        save_prefix=None,
        groupby="compartment_narrow",
        dot_min=0.1,
        dot_max=1,
        log=True,
        standard_scale="var",
        **kwargs,
    ):
        """
        Make one dotplot of all genes, per species
        """
        # Return groups of up to 50 genes at a time
        genes_collated = grouper(genes, max_genes)
        for i, genes_subset in enumerate(genes_collated):
            print(f"gene subset #{i + 1}")
            genes_subset_no_none = [x for x in genes_subset if x is not None]

            for species, ad in self.species_adatas.items():
                print(species)
                color_map = self.species_to_color_map[species]
                if save_prefix:
                    save = \
                        f"{save_prefix}__genesubset-{i}__species-{species}.png"
                    show = False
                else:
                    save = None
                    show = True
                sc.pl.dotplot(
                    ad,
                    genes_subset_no_none,
                    groupby=groupby,
                    standard_scale=standard_scale,
                    log=log,
                    color_map=color_map,
                    dot_min=dot_min,
                    dot_max=dot_max,
                    save=save,
                    show=show,
                    **kwargs,
                )

    def de_dotplots(
        self,
        sort="alphabetically",
        select_genes=None,
        ignore_genes=None,
        primate_larger=True,
        save=False,
        save_format="png",
        **kwargs,
    ):
        """ "Make dotplots for differentially expressed genes, one per celltype

        sort : str
            Whether to sort gene names alphabetically or by score value
        primate_larger : bool
            If True, choose genes that are more highly expressed in both primates.
            If False, choose genes that are less expressed in both primates
        """
        for group, ad in self.mini_adatas.items():
            self.print_group(group)

            # Get primate-differentially expressed genes
            gene_names = self.get_differentially_expressed_gene_names(
                ad,
                sort,
                ignore_genes=ignore_genes,
                select_genes=select_genes,
                primate_larger=primate_larger,
            )

            if gene_names:
                if save:
                    save_suffix = (
                        sanitize(group)
                        + f"__primate_larger-{primate_larger}.{save_format}"
                    )
                else:
                    save_suffix = None

                if primate_larger:
                    title = f"{group}\nPrimate gain of expression"
                else:
                    title = f"{group}\nPrimate loss of expression"

                sc.pl.dotplot(
                    ad,
                    gene_names,
                    groupby=self.species_col,
                    title=title,
                    save=save_suffix,
                    **kwargs,
                )
            else:
                print(f"No genes found with primate_larger={primate_larger}")

    def dotplots(
        self,
        gene_names,
        select_group=None,
        save=False,
        save_format="png",
        max_genes=55,
        **kwargs,
    ):
        """ "Make dotplots for differentially expressed genes, one per celltype

        sort : str
            Whether to sort gene names alphabetically or by score value
        primate_larger : bool
            If True, choose genes that are more highly expressed in both primates.
            If False, choose genes that are less expressed in both primates
        """
        # Return groups of up to 50 genes at a time
        genes_collated = grouper(gene_names, max_genes)
        for i, genes_subset in enumerate(genes_collated):
            print(f"gene subset #{i + 1}")
            genes_subset_no_none = [x for x in genes_subset if x is not None]

            for group, ad in self.mini_adatas.items():
                if select_group is not None:
                    # Use "in" instead of "==" to allow for
                    # select_group="artery cell" and group="endothelial: artery cell"
                    if select_group != group:
                        continue
                self.print_group(group)

                if save:
                    save_suffix = (
                        sanitize(group) + f"{save}__genesubset-{i}.{save_format}"
                    )
                else:
                    save_suffix = None

                sc.pl.dotplot(
                    ad,
                    genes_subset_no_none,
                    groupby=self.species_col,
                    save=save_suffix,
                    title=group,
                    **kwargs,
                )

    def de_dotplots_multispecies(
        self,
        sort="score",
        save=True,
        abs_score_threshold=1e-3,
        ignore_genes=None,
        select_genes=None,
        **kwargs,
    ):
        """ "Make dotplots for differentially expressed geens

        sort : str
            Whether to sort gene names alphabetically or by score value
        """
        for group, ad in self.mini_adatas.items():
            self.print_group(group)

            # Get primate-differentially expressed genes
            gene_names = self.get_differentially_expressed_gene_names(
                ad, sort, select_genes=select_genes, ignore_genes=ignore_genes
            )

            print(f"Total number of primate-enriched genes: {len(gene_names)}")
            if save:
                save_prefix = sanitize(group)
            else:
                save_prefix = False

            self.dotplot_multispecies(gene_names, save_prefix=save_prefix, **kwargs)

    def heatmaps(self, sort="alphabetically", log=True, **kwargs):
        """ 

        sort : str
            Whether to sort gene names alphabetically or by score value
        """
        for group, ad in self.mini_adatas.items():
            self.print_group(group)

            gene_names = self.get_differentially_expressed_gene_names(ad, sort)
            sc.pl.heatmap(ad, gene_names, groupby=self.species_col, log=log, **kwargs)

    def violinplots(self, sort="alphabetically", log=True, **kwargs):
        """ "


        sort : str
            Whether to sort gene names alphabetically or by score value
        """
        for group, ad in self.adatas.items():
            self.print_group(group)


            gene_names = get_differentially_expressed_gene_names(ad, sort)
            sc.pl.heatmap(
                ad, gene_names, groupby=self.species_col, log=log, **kwargs)

    def plot_de_results(
            self, sort="alphabetically", subgroup_col=None, **kwargs):
        """"

        sort : str
            Whether to sort gene names alphabetically or by score value
        """
        ontology_species_col = "ontology_species"

        for group, ad in self.mini_adatas.items():
            self.print_group(group)

            gene_names = self.get_differentially_expressed_gene_names(ad, sort)

            # Groupby species plots
            sc.pl.dotplot(ad, gene_names, groupby=self.species_col)
            sc.pl.heatmap(ad, gene_names, groupby=self.species_col, log=True)
            sc.pl.umap(ad, color=gene_names)

            if subgroup_col is not None:
                # Have not run or tested..
                # Set species to specific order
                terms = [
                    f"{cell_ontology_class} ({species})"
                    for cell_ontology_class, species in zip(
                        ad.obs[subgroup_col].values, ad.obs.species.values
                    )
                ]
                ad.obs[ontology_species_col] = pd.Categorical(
                    terms, categories=ontology_species_order, ordered=True
                )
                # Make violinplots within subgroup
                # Make colors for matrix plot
                ontology_to_color = pd.Series(
                    ad.uns["cell_ontology_class_colors"],
                    index=ad.obs.cell_ontology_class.cat.categories,
                )
                ontology_species_to_color = pd.Series(
                    dict(
                        zip(
                            ad.obs[ontology_species_col],
                            ontology_to_color[
                                ad.obs[
                                    ontology_species_col].str.split(
                                        r" (").str[0]
                            ],
                        )
                    )
                )
                ontology_species_to_color = \
                    ontology_species_to_color.sort_index()

                sc.pl.stacked_violin(
                    ad,
                    median_scores.index,
                    groupby=ontology_species_col,
                    row_palette=ontology_species_to_color,
                )

    def do_differential_expression(
        self,
        method="logreg",
        use_raw=True,
        penalty="l1",
        solver="saga",
        class_weight="balanced",
        reference="ref_species",
        **kwargs,
    ):
        if reference == "ref_species":
            reference = self.ref_species
        for group, ad in self.mini_adatas.items():
            self.print_group(group)
            sc.tl.rank_genes_groups(
                ad,
                groupby=self.species_col,
                reference=reference,
                method=method,
                use_raw=use_raw,
                penalty=penalty,
                solver=solver,
                class_weight=class_weight,
                **kwargs,
            )
            sc.pl.rank_genes_groups(ad)

    @staticmethod
    def print_group(group):
        print(f"\n------- group: {group} -------")

    def count_per_species_cell_types(
        self, groupby=["species", "compartment_group", "narrow_group"]
    ):
        counts = self.adata_original.obs.groupby(groupby).size()
        counts.name = "n_cells"
        counts = counts.reset_index()
        print(counts.shape)
        return counts

    def celltype_barplot(
        self,
        cell_type_counts,
        cell_group="narrow_group",
        groupby="compartment_group",
        fig_width=1.5,
        fig_height_scale=0.3,
        context="paper",
        style="whitegrid",
    ):
        sns.set(context=context, style=style)

        for compartment, df in cell_type_counts.groupby(groupby):
            if compartment == "nan":
                continue
            print(f"--- compartment: {compartment} ---")
            fig_height = df[cell_group].nunique() * fig_height_scale
            aspect = fig_width / fig_height
            df[cell_group] = df[cell_group].astype(str)
            df[groupby] = df[groupby].astype(str)

            # Replace non-present cell types with 0 for consistent plotting
            df_fillna = (
                df.set_index(self.species_col)
                .groupby([groupby, cell_group])
                .apply(lambda x: x.loc[plot_constants.SPECIES_ORDER, "n_cells"].fillna(0))
            )  # .reset_index(-1)
            df_fillna = df_fillna.stack()
            df_fillna.name = "n_cells"
            df_fillna = df_fillna.reset_index()

            g = sns.catplot(
                data=df_fillna,
                y=cell_group,
                x="n_cells",
                hue=self.species_col,
                hue_order=plot_constants.SPECIES_ORDER,
                palette=SPECIES_PALETTE,
                height=fig_height,
                aspect=aspect,
                kind="bar",
                col=groupby,
                legend=False,
                zorder=-1,
            )
            g.set_titles("{col_name}")
            g.set(xscale="log", xlim=(0, 1e4), ylabel=None)
            for ax in g.axes.flat:
                ax.set(xticks=[1e1, 1e2, 1e3, 1e4])
                ax.grid(
                    axis="x", color="white", linestyle="-",
                    linewidth=1, zorder=1)
                # ax.axvline(0, y)

    def plot_shared_cell_types(self, fig_height_scale=0.3):
        counts = self.count_per_species_cell_types()
        self.celltype_barplot(counts, fig_height_scale=fig_height_scale)

    def make_diffexpr_info_table(self, adata, dissociation_genes=None):

        scores = self.extract_de_scores(adata)

        logfoldchanges = self.extract_de_logfoldchanges(adata)

        pvals_adj = self.extract_de_pvals_adj(adata)

        gene_names = self.extract_de_names(adata)
        
        try:
            de_percent_expressing = self.extract_de_percent_expressing(adata)
            de_percent_expressing.columns = pd.MultiIndex.from_product([['percent_expressing'], de_percent_expressing.columns.tolist()])
        except KeyError:
            de_percent_expressing = None

        logfoldchanges_series = logfoldchanges.unstack()
        logfoldchanges_series.name = "logfoldchange"

        gene_names_series = gene_names.unstack()
        gene_names_series.name = "gene_name"

        pvals_adj_series = pvals_adj.unstack()
        pvals_adj_series.name = "pval_adj"

        neg_log10_pvals_adj_series = -np.log10(pvals_adj_series)
        neg_log10_pvals_adj_series.name = "pval_adj_neg_log10"

        scores_series = scores.unstack()
        scores_series.name = "score"

        de_info = pd.concat(
            [
                logfoldchanges_series,
                gene_names_series,
                pvals_adj_series,
                neg_log10_pvals_adj_series,
                scores_series,
            ],
            axis=1,
        ).reset_index(level=0)
        de_info = de_info.rename(columns={"level_0": "species"})
        de_info.head()

        de_info_2d = de_info.pivot_table(
            index="gene_name",
            columns=["species"],
            values=["logfoldchange", "pval_adj", "pval_adj_neg_log10", "score"],
        )
#         import pdb; pdb.set_trace()
        de_info_2d = pd.concat([de_info_2d, de_percent_expressing], axis=1)
        if dissociation_genes is not None:
            de_info_2d["is_dissociation_gene"] = de_info_2d.index.isin(
                dissociation_genes
            )
        return de_info_2d

    @staticmethod
    def filter_diffexpr_info(de_info_2d):

        same_directional_change = (de_info_2d.logfoldchange < 0).all(axis=1) | (
            de_info_2d.logfoldchange > 0
        ).all(axis=1)
        same_directional_change.sum()

        de_filter = (de_info_2d.pval_adj < 0.05).all(axis=1) & same_directional_change

        de_info_2d_filtered = de_info_2d.loc[de_filter, :]
        return de_info_2d_filtered

    def get_per_group_diffexpr_tables(self, dissociation_genes=None):
        for group, ad in self.mini_adatas.items():
            self.print_group(group)
            diffexpr_info = self.make_diffexpr_info_table(
                ad, dissociation_genes=dissociation_genes
            )
            yield group, diffexpr_info


class PerGroupAgg:
    def __init__(
        self,
        adata,
        group,
        species="species_batch",
        species_order=plot_constants.SPECIES_BATCH_ORDER,
        compartment_narrow_separator=" - ",
    ):
        self.adata = adata
        self.group = group
        self.species = species
        self.species_order = species_order
        self.groupby = [self.species, self.group]
        self.groupby_sizes = self.get_groupby_sizes()
        self.compartment_narrow_separator = compartment_narrow_separator

    def get_groupby_sizes(self):
        groupby_sizes = self.adata.obs.groupby(self.groupby).size()
        groupby_sizes.name = f"Number of cells per {self.group}"
        return groupby_sizes

    @staticmethod
    def aggregate(X, aggfunc="mean", **kws):
        if aggfunc == "mean":
            aggregated = np.ravel(X.mean(axis=0, dtype=np.float64))

            # callable = "isinstance(x, function)"
            # Check if aggfunc is a python function
        elif callable(aggfunc):
            # Sparse matrices don't have a native median function,
            # have to convert to dense matrix
            aggregated = np.ravel(aggfunc(X.todense(), axis=0, **kws))
        else:
            raise ValueError(
                f"Aggregation function '{aggfunc}' is not supported!"
                "Only 'mean' and 'median' are"
            )
        return aggregated

    @staticmethod
    def get_x(adata, layer=None):
        if layer is not None:
            return adata.layers[layer]
        else:
            return adata.X

    @staticmethod
    def new_idx(adata, idx, gene_symbols):
        if gene_symbols is not None:
            return adata.var[idx]
        else:
            return adata.var_names

    def grouped_obs_agg(
        self, adata, groupby, layer=None, gene_symbols=None,
        aggfunc="mean", **kws
    ):

        grouped = adata.obs.groupby(groupby)
        out = pd.DataFrame(
            np.zeros((adata.shape[1], len(grouped)), dtype=np.float64),
            columns=list(grouped.groups.keys()),
            index=adata.var_names,
        )

        for group, idx in grouped.indices.items():
            X = self.get_x(adata[idx])
            out[group] = self.aggregate(X, aggfunc, **kws)

        # If groupby is list or tuple, then make the columns a MultiIndex
        if isinstance(groupby, (list, tuple)):
            out.columns = pd.MultiIndex.from_tuples(out.columns)
        return out

    @staticmethod
    def sanitize_aggfunc(aggfunc, aggfunc_name):
        if isinstance(aggfunc, str):
            return aggfunc
        elif aggfunc_name is not None:
            return aggfunc_name
        else:
            raise ValueError("aggfunc_name is None! Cannot rename or sanitize")

    @staticmethod
    def get_n_nonzero(aggregated, aggfunc_name):
        n_aggregated_nonzero = (aggregated > 0).sum()
        n_aggregated_nonzero.name = \
            f"Number of genes with nonzero {aggfunc_name}"
        return n_aggregated_nonzero

    def do_aggregation(
        self,
        layer=None,
        gene_symbols=None,
        aggfunc="mean",
        aggfunc_name=None,
        plot=False,
        **kws,
    ):
        aggfunc_name = self.sanitize_aggfunc(aggfunc, aggfunc_name)

        aggregated = self.grouped_obs_agg(
            self.adata,
            self.groupby,
            layer=layer,
            gene_symbols=gene_symbols,
            aggfunc=aggfunc,
            **kws,
        )
        self.aggregated = aggregated

        n_aggregated_nonzero = self.get_n_nonzero(aggregated, aggfunc_name)
        self.n_aggregated_nonzero = n_aggregated_nonzero
        self.aggregated_nonzero = aggregated.loc[(aggregated > 0).any(axis=1)]
        self.tidy_nonzero = self.tidy_n_nonzero_vs_size()
        if plot:
            self.plot_nonzero_agg_vs_groupby_sizes()

    def tidy_n_nonzero_vs_size(self):
        nonzero = self.n_aggregated_nonzero
        sizes_df = self.groupby_sizes.to_frame()
        nonzero_df = nonzero.to_frame()

        df = pd.concat([sizes_df, nonzero_df], axis=1)
        df = df.reset_index()
        df = df.rename(
            columns={"level_0": self.species, "level_1": self.group})
        return df

    def plot_nonzero_agg_vs_groupby_sizes(
            self, methods=("spearman", "pearson")):
        nonzero = self.n_aggregated_nonzero
        tidy = self.tidy_nonzero

        x = self.groupby_sizes.name
        y = nonzero.name

        for method in methods:
            corr = tidy[x].corr(tidy[y], method=method)
            print(
                f"{method.capitalize()} correlation of"
                "sizes vs nonzero: {corr:.3f}")

        g = sns.FacetGrid(data=tidy, hue=self.species)
        g.map(sns.scatterplot, x, y)
        g.add_legend()

        g = sns.FacetGrid(data=tidy, hue=self.group, aspect=2)
        g.map(sns.scatterplot, x, y)
        g.add_legend()

    def bootstrap_average(
        self,
        aggfunc="mean",
        n_iteration=1000,
        n_per_group=1000,
        plot=True,
    ):
        dfs = []

        for iteration in tqdm(range(n_iteration)):
            #     iteration = 0
            obs_subsampled = self.adata.obs.groupby(
                self.groupby, as_index=False, group_keys=False
            ).apply(
                lambda x: x.sample(
                    n_per_group, random_state=2020 + iteration, replace=True
                )
            )
            adata_subsampled = adata_same_narrow_group[obs_subsampled.index]
            df = compartment_agg.grouped_obs_agg(
                adata_subsampled, groupby, aggfunc=aggfunc
            )
            dfs.append(df)

        means = pd.concat(dfs, axis=1)
        print(means.shape)
        means.head()

        self.mean_of_means = means.groupby(level=[0, 1], axis=1).mean()
        self.aggregated = self.mean_of_means
        mean_of_means_nonzero = self.get_n_nonzero(self.mean_of_means, key)
        self.aggregated_nonzero = mean_of_means_nonzero
        self.tidy_nonzero = self.tidy_n_nonzero_vs_size()

        if plot:
            self.plot_nonzero_agg_vs_groupby_sizes(aggfunc)

    def do_correlation(self, method="spearman"):
        self.method = method
        self.correlated = self.aggregated.corr(method=method)
        self.correlated_tidy = self.make_tidy_correlation(self.correlated)

    def make_tidy_correlation(self, data2d):
        is_multiindex = isinstance(data2d.index, pd.MultiIndex)

        if is_multiindex:
            level = [0, 1]
            renamer = {
                "level_0": "species1",
                "level_1": "compartment_narrow1",
                "level_2": "species2",
                "level_3": "compartment_narrow2",
                0: f"{self.method}_r",
            }
        else:
            level = -1
            renamer = {
                data2d.columns.name: "compartment_narrow1",
                "level_0": "compartment_narrow2",
                0: data2d.columns.name,
            }
        tidy = data2d.stack(level=level)
        tidy = tidy.reset_index()
        # if is_multiindex:
        #     tidy = tidy.drop("level_4", axis=1)
        tidy = tidy.rename(columns=renamer)
        if is_multiindex:
            tidy = tidy.query("species1 != species2")
        else:

            tidy = tidy.query('compartment_narrow1 == compartment_narrow2')
            tidy = tidy.drop('compartment_narrow2', axis=1)
            tidy = tidy.rename(
                columns={'compartment_narrow1': 'compartment_narrow'})
            tidy = tidy.sort_values(data2d.columns.name)
        return tidy

    def make_species_centric_correlation(self, species1):
        correlation_species1 = self.correlated_tidy.query(
            "species1 == @species1")
        correlation_species1 = correlation_species1.drop(
            "compartment_narrow2", axis=1)
        correlation_species1 = correlation_species1.rename(
            columns={"compartment_narrow1": "compartment_narrow"}
        )
        return correlation_species1

    def do_correlation_difference(self, anchor_species):
        """Assumes there are only three species total"""
        self.anchor_species = anchor_species
        self.other_species = [
            x for x in self.species_order if x != anchor_species]
        name = (
            f"corr({anchor_species}, {self.other_species[0]}) - "
            f"corr({anchor_species}, {self.other_species[1]})"
        )

        corr_dfs = []

        for other in self.other_species:
            other_corr = self.correlated.loc[anchor_species][other]
            corr_dfs.append(other_corr)

        diff = corr_dfs[0].subtract(corr_dfs[1])
        diff.columns.name = name
        self.correlation_difference = diff
        self.correlation_difference_tidy = self.make_tidy_correlation(diff)

    def get_same_celltype_correlation_diff(self):
        self.correlation_diff_same_celltype = \
            self.correlation_difference.query(
                "compartment_narrow1 == compartment_narrow2")

    def correlation_heatmap(self):
        corr = self.correlated.loc[self.anchor_species, self.other_species]
        corr_by_celltype = corr.sort_index(axis=1, level=1)
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.heatmap(corr_by_celltype, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set(title=f"{self.method} correlation")

    def correlation_difference_plot(self):
        fig, ax = plt.subplots(figsize=(4, 3))

        colors = [
            compartment_to_color[x.split(self.compartment_narrow_separator)[0]]
            for x in self.correlation_difference_tidy.compartment_narrow
        ]

        vmax = (
            self.correlation_difference_tidy[self.correlation_difference.columns.name]
            .abs()
            .max()
        )

        sns.barplot(
            y="compartment_narrow",
            x=self.correlation_difference.columns.name,
            data=self.correlation_difference_tidy,
            palette=colors,
        )
        sns.despine()
        title = self._correlation_difference_title(self.anchor_species)
        ax.set(
            xlim=(-vmax, vmax),
            # xticks=np.arange(-vmax, vmax, step=0.1),
            title=title,
            xlabel=f"$\\Delta$ {self.method} correlation",
            ylabel=None,
        )
        ax.axvline(0, color="black")
        plt.xticks(rotation=90)
        fig.tight_layout()
        # median_outdir = f"{outdir_gdrive}/correlation_difference/"
        # fig.savefig(
        #     f"{median_outdir}/human_centric_difference_of_median_correlation.pdf"
        # )

    def _correlation_difference_title(self, anchor_species):
        other_species = [x for x in self.species_order if x != anchor_species]
        left = f"{anchor_species}:{other_species[1]}"
        right = f"{anchor_species}:{other_species[0]}"
        title = left + r" $\leftarrow$ $\rightarrow$ " + right
        return title


def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks

    grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"

    Cribbed from
    https://docs.python.org/3/library/itertools.html#itertools-recipes
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)



def sanitize(x):
    return (
        x.replace("/", "-slash-")
        .replace(": ", "--")
        .replace(" ", "_")
        .replace(":", "--")
        .replace(",", "_")
        .replace("=", "-eq-")
        .lower()
    )


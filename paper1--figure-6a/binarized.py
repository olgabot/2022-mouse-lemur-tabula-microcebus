"""
Code to analyze patterns of on/off gene expression across mouse, mouse lemur, and human
"""

import itertools
import logging
import math
import os

# Joblib for parallelizing
from joblib import Parallel, delayed
import numpy as np
import matplotlib_venn
import matplotlib.pyplot as plt
import mygene
import pandas as pd
import scanpy as sc
import scipy
from scipy.stats import hypergeom
import seaborn as sns
from sklearn.metrics import normalized_mutual_info_score
from tqdm import tqdm

MG = mygene.MyGeneInfo()

# Ignore "too many figures open" warning
plt.rcParams.update({"figure.max_open_warning": 0})

# local modules
import enrich

# Create a logger
logging.basicConfig(format="-- %(name)s --\n%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

SPECIES_ORDER = ["Mouse", "Mouse lemur", "Human"]
SPECIES_ORDER_HUMAN_FIRST = SPECIES_ORDER[::-1]
SPECIES_BATCH_ORDER = ["mouse", "lemur", "human"]
SPECIES_BATCH_ORDER_HUMAN_FIRST = SPECIES_BATCH_ORDER[::-1]

N_SPECIES = len(SPECIES_ORDER)
SPECIES_PALETTE = sns.color_palette(n_colors=N_SPECIES)
SPECIES_PALETTE_HUMAN_FIRST = SPECIES_PALETTE[::-1]

# sns.palplot(SPECIES_PALETTE)

GROUP_COLS = ["narrow_group", "broad_group", "compartment_group"]
NARROW_GROUP = "narrow_group"

OUTDIR_GDRIVE = "/home/olga/googledrive/TabulaMicrocebus/data/cross-species/binarized"
OUTDIR_LOCAL = (
    "/home/olga/data_sm/tabula-microcebus/data-objects/cross-species/binarized"
)
OUTDIRS = OUTDIR_LOCAL, OUTDIR_GDRIVE

PUBMED_TEMPLATE = "https://pubmed.ncbi.nlm.nih.gov/{pubmed_id}/"


def get_nrow_ncol(n_items):
    n_col = int(math.ceil(math.sqrt(n_items)))
    n_row = 1
    while n_row * n_col < n_items:
        n_row += 1
    return n_row, n_col


def maybe_make_directories(directories):
    # Make the output directories if they don't exist already
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)


def turn_off_empty_axes(axes):
    for ax in axes.flat:
        ax_has_stuff = ax.lines or ax.collections
        if not ax_has_stuff:
            ax.axis("off")


class BinarizedAnalysesBase:
    def __init__(
        self,
        adata,
        groupby,
        species_col,
        gene_subset=None,
        most_specific_group=NARROW_GROUP,
        group_cols=GROUP_COLS,
        debug=False,
        min_fraction=0.05,
        outdirs=OUTDIRS,
        min_group_size=20,
        species_order=SPECIES_BATCH_ORDER_HUMAN_FIRST,
        species_palette=SPECIES_PALETTE_HUMAN_FIRST,
    ):
        self.adata = adata
        self.groupby = groupby
        self.species = species_col
        self.most_specific_group = most_specific_group
        self.group_cols = group_cols
        self.debug = debug
        self.min_fraction = min_fraction
        self.outdirs = outdirs
        self.outdirs_groupby = [
            os.path.join(outdir, self.groupby) for outdir in self.outdirs
        ]
        maybe_make_directories(self.outdirs)
        maybe_make_directories(self.outdirs_groupby)

        # Minimum nubmer of cells per group
        self.min_group_size = min_group_size

        # Make sure as many colors are specified as species
        assert len(species_order) == len(species_palette)
        self.species_order = species_order
        self.species_palette = species_palette
        self.n_species = len(species_order)

        if debug:
            logger.setLevel(logging.DEBUG)

        self.gene_lists = {}
        self.compute_expressed_fractions(
            min_fraction=min_fraction, gene_subset=gene_subset
        )
        self.binarized = self._binarize()

        # This is set after the super init so the group fractions are already filtered
        # for non-expressed genes
        self.gene_lists["background"] = list(self.group_fractions.columns)
        self.n_genes = len(self.gene_lists["background"])

        # For functional gene enrichment, e.g. gene ontology
        self.enrichments = {}

    def compute_expressed_fractions(self, min_fraction=None, gene_subset=None):
        # TODO: Pycharm tells me it's not good practice to put all these attribute
        #  assignments in a function. So what's a better way to do it?
        logger.info(f"Computing fraction cells expressing genes per {self.groupby}")
        self.adata_shared = self._get_adata_shared_celltypes()
        fractions = self._fraction_expression_per_group()

        if gene_subset is not None:
            fractions = fractions[gene_subset]

        fractions = self._remove_groups_not_present_in_all_species(
            fractions, self.n_species
        )
        self.group_fractions = self._remove_genes_not_expressed_anywhere(fractions)
        self.plot_fractions(min_fraction=min_fraction)

    def _binarize(self):
        binarized = self.group_fractions > self.min_fraction
        # TODO: there's more to filter here
        return binarized

    def _get_adata_shared_celltypes(self):
        """Use only cell types shared across all three species"""
        logger.info(
            f"Filtering adata for only {self.most_specific_group} shared "
            f"across all {self.n_species} species"
        )
        logger.info(
            f"Starting number of {self.most_specific_group}: "
            f"{self.adata.obs[self.most_specific_group].nunique()}"
        )
        grouped = self.adata.obs.groupby(self.most_specific_group)
        obs_shared = grouped.filter(
            lambda x: x[self.species].nunique() == self.n_species
        )
        logger.info(
            f"After removing groups not in all species {self.most_specific_group}: "
            f"{obs_shared[self.most_specific_group].nunique()}"
        )
        # Filter for groups with minimum number of cells
        obs_share_enough_cells = obs_shared.groupby(self.groupby).filter(
            lambda x: len(x) >= self.min_group_size
        )
        obs_share_enough_cells = obs_share_enough_cells.dropna(subset=self.group_cols)

        logger.info(
            f"After cell groups with too few cells in {self.groupby}: "
            f"{obs_share_enough_cells[self.groupby].nunique()}"
        )

        adata_shared = self.adata[obs_share_enough_cells.index]
        logger.info(
            f"Filtered number of *shared* {self.most_specific_group}: "
            f"{adata_shared.obs[self.most_specific_group].nunique()}"
        )
        return adata_shared

    def _fraction_expression_per_group(self):
        # Assign here for shorter line lengths
        logger.info(f"Computing expressed fractions across shared {self.groupby}")
        adata = self.adata_shared
        group_fractions = adata.obs.groupby([self.species, self.groupby]).apply(
            lambda x: self._get_fraction_cells_expressing(x, adata)
        )
        return group_fractions

    @staticmethod
    def _get_fraction_cells_expressing(df, adata):
        mini_adata = adata[df.index, :]
        n_cells_expressing = (mini_adata.X > 0).sum(axis=0).A1
        fraction_cells = n_cells_expressing / mini_adata.X.shape[0]
        series = pd.Series(fraction_cells, index=adata.var.index)
        return series

    def plot_fractions(self, min_fraction=None):
        fig, ax = plt.subplots()

        sns.distplot(self.group_fractions.values.flat)
        if min_fraction is not None:
            ax.axvline(min_fraction, color="red")
        sns.despine()
        ax.set(title=f"Fraction cells expressing a gene in {self.groupby}")

    @staticmethod
    def _remove_genes_not_expressed_anywhere(fractions):
        cols = (fractions > 0).any()
        fractions_no_not_expressed = fractions.loc[:, cols]
        logger.debug(
            f"fractions_no_not_expressed.shape: " f"{fractions_no_not_expressed.shape}"
        )
        return fractions_no_not_expressed

    @staticmethod
    def _remove_groups_not_present_in_all_species(fractions, n_species):
        fractions_all_species = (
            fractions.dropna()
            .groupby(level=1, axis=0)
            .filter(lambda x: len(x) == n_species)
        )
        logger.debug(f"fractions_all_species.shape: {fractions_all_species.shape}")
        return fractions_all_species

    def venn(self, gene_list, figsize=None, n_row=None, n_col=None, annotate=True):
        gene_names = self.gene_lists[gene_list]
        grouped = self.binarized[gene_names].groupby(level=1, observed=True)

        if n_row is None and n_col is None:
            n_items = len(grouped.groups)
            n_row, n_col = get_nrow_ncol(n_items)

        if figsize is None:
            # (16, 8) for 5 items, 3 columns, 2 rows
            figwidth = n_col * 4
            figheight = n_row * 3
            figsize = (figwidth, figheight)

        fig, axes = plt.subplots(ncols=n_col, nrows=n_row, figsize=figsize)

        for ax, (group_name, df) in zip(axes.flat, grouped):
            # print(f"--- group_name: {group_name} ---")
            if df.empty:
                continue

            set_list = []
            for species in self.species_order:
                set_list.append(set(df.columns[df.loc[species].values.flatten()]))

            total = len(set_list[0].union(set_list[1]).union(set_list[2]))
            if annotate:
                subset_label_formatter = lambda x: f"{x} ({(x/total):1.0%})"
            else:
                subset_label_formatter = None

            matplotlib_venn.venn3(
                set_list,
                set_labels=self.species_order,
                set_colors=self.species_palette,
                ax=ax,
                subset_label_formatter=subset_label_formatter,
            )

            title = f"{group_name}\n(n = {total} genes)"
            ax.set_title(title)

        turn_off_empty_axes(axes)

        fig.suptitle(gene_list)
        fig.tight_layout()

    def _log_n_genes_and_percentage(self, n_subgroup, subgroup_name):
        percentage = 100 * n_subgroup / self.n_genes
        logger.info(
            f"Number of {subgroup_name} genes: {n_subgroup}/{self.n_genes} "
            f"({percentage:.2f}%)"
        )

    def do_go_enrichment(
        self, plot=False, gene_ontology_only=False, groupby_source=False, write=False
    ):
        background = self.gene_lists["background"]
        for name, gene_list in self.gene_lists.items():
            if name == "background":
                continue

            enrichment = enrich.do_go_enrichment(list(gene_list), background=background)
            self.enrichments[name] = enrichment
            if enrichment.empty:
                logger.info(f"Functional enrichment for {name} is empty! Skipping...")
                continue

            title = name.capitalize()
            if not enrichment.empty and plot:
                self._single_plot_enrichment(
                    enrichment, f"{title} genes", gene_ontology_only, groupby_source
                )
            if not enrichment.empty and write:
                for d in self.outdirs_groupby:
                    csv = (
                        f"{d}/binarized_expression__stable_genes__{name}"
                        f"__{self.groupby}__functional_enrichment.csv"
                    )
                    enrichment.to_csv(csv)
                    logger.info(f"Wrote:\n{csv}")

    def plot_go_enrichment(self, gene_ontology_only=False, groupby_source=False):
        for name, gene_list in self.gene_lists.items():
            if name == "background":
                continue

            title = name.capitalize()
            enrichment = self.enrichments[name]
            if not enrichment.empty:
                self._single_plot_enrichment(
                    enrichment, f"{title} genes", gene_ontology_only, groupby_source
                )
            else:
                logger.info(f"{name} enrichment is empty")

    def _single_plot_enrichment(
        self, enrichment, title, gene_ontology_only=False, groupby_source=False
    ):
        if groupby_source and gene_ontology_only:
            raise ValueError(
                "Cannot specify both gene_ontology_only and "
                "groupby_source! Only one can be true"
            )
        if groupby_source:
            # Make multiple plots, one for each enrichment dataset source
            for source, df in enrichment.groupby("source"):
                enrich.plot_go_enrichment(df)
                ax = plt.gca()
                source_title = f"{title} (Enrichment source: {source})"
                ax.set(title=source_title)
                self._save_enrichment_fig(source_title)
        else:
            if gene_ontology_only:
                title += " (Gene Ontology only)"
                enrichment = enrichment[enrichment.source.str.contains("GO")]
            enrich.plot_go_enrichment(enrichment)
            ax = plt.gca()
            ax.set(title=title)
            self._save_enrichment_fig(title)

    @staticmethod
    def sanitize(x):
        return x.replace("/", "-slash-").replace(" ", "_").replace(":", "--").lower()

    def _save_enrichment_fig(self, title):
        title_sanitized = self.sanitize(title)

        fig = plt.gcf()
        for outdir in self.outdirs_groupby:
            pdf = (
                f"{outdir}/binarized_expression__"
                f"__{self.groupby}__{title_sanitized}__functional_enrichment.pdf"
            )
            fig.tight_layout()
            fig.savefig(pdf)

    def plot_venns(self):
        for name, gene_list in self.gene_lists.items():
            if name == "background" or name == "conserved_all":
                continue
            self.venn(name)


class BinarizedStabilityAnalyses(BinarizedAnalysesBase):
    """Find genes whose on/off signal is stable across a range of thresholds"""

    def __init__(
        self,
        adata,
        groupby,
        species_col,
        most_specific_group=NARROW_GROUP,
        group_cols=GROUP_COLS,
        debug=False,
        species_order=SPECIES_ORDER_HUMAN_FIRST,
        species_palette=SPECIES_PALETTE_HUMAN_FIRST,
    ):
        super().__init__(
            adata,
            groupby,
            species_col,
            most_specific_group=most_specific_group,
            group_cols=group_cols,
            debug=debug,
            species_order=species_order,
            species_palette=species_palette,
        )

    def do_stability_analysis(
        self,
        thresholds=np.arange(0.01, 0.21, 0.01),
        min_mutual_information=0,
        n_jobs=16,
    ):
        # TODO: Pycharm tells me it's not good practice to put all these attribute
        #  assignments in a function. So what's a better way to do it?
        self._binarizeds = self._binarize_many_thresholds(thresholds)
        self.mutual_information = self._compute_mutual_information(n_jobs=n_jobs)
        self._plot_mutual_information_distribution(min_mutual_information)

        self.mutual_information_stable = self._filter_stable_genes()
        self._set_stable_gene_names()
        self._set_unstable_gene_names()
        self.plot_venns()

    def _binarize_many_thresholds(self, min_fractions=np.arange(0.01, 0.21, 0.01)):
        binarizeds = {}

        for min_fraction in tqdm(
            min_fractions, desc=f"Binarizing at thresholds: {min_fractions.tolist()}"
        ):
            # Skip 0 because it skews the distribution
            if min_fraction == 0:
                continue

            binarized = self.group_fractions > min_fraction

            # Stringify the key to avoid weird rounding errors
            key = f"{100 * min_fraction:.2f}%"
            binarizeds[key] = binarized
        return binarizeds

    def _compute_mutual_information(self, n_jobs=16):
        logger.info("Computing pairwise mutual information across thresholds")
        iterator = itertools.combinations(self._binarizeds.items(), 2)
        dfs = Parallel(n_jobs=n_jobs, verbose=1)(
            delayed(do_per_column_mutual_information)(group1, group2)
            for group1, group2 in tqdm(
                iterator, desc="Computing mutual information between thresholds"
            )
        )
        mutual_information_df = pd.concat(dfs)
        mutual_information_df = mutual_information_df.rename(
            columns={"index": "gene_name"}
        )
        logger.debug(f"mutual_information_df.shape: {mutual_information_df.shape}")
        return mutual_information_df

    def _plot_mutual_information_distribution(self, min_mutual_information=0):
        fig, ax = plt.subplots()

        sns.distplot(self.mutual_information.mutual_information)
        ax.axvline(min_mutual_information, color="red")
        ax.set(title="Mutual Information across all threshold pairs")
        sns.despine()

    def _filter_stable_genes(self):
        mutual_information_df_stable = self.mutual_information.groupby(
            "gene_name"
        ).filter(lambda x: (x.mutual_information > 0.0).all())
        print(mutual_information_df_stable.shape)
        mutual_information_df_stable.head()
        return mutual_information_df_stable

    def _set_stable_gene_names(self):
        self.stable_gene_names = self.mutual_information_stable.gene_name.unique()
        self.gene_lists["stable"] = list(self.stable_gene_names)
        self.n_stable = len(self.stable_gene_names)
        self._log_n_genes_and_percentage(self.n_stable, "stable")

    def _set_unstable_gene_names(self):
        self.unstable_gene_names = self.group_fractions.columns.difference(
            self.stable_gene_names
        )
        self.gene_lists["unstable"] = list(self.unstable_gene_names)
        self.n_unstable = len(self.unstable_gene_names)
        self._log_n_genes_and_percentage(self.n_unstable, "unstable")

    def write_csvs(self):
        for d in self.outdirs:
            [stable_gene_names].to_csv(
                f"{d}/{self.groupby}__binarized__stable_genes__fractions.csv"
            )
        stable_genes_series = pd.Series(self.stable_gene_names)
        stable_genes_series = stable_genes_series.sort_values()
        for d in self.outdirs:
            stable_genes_series.to_csv(
                f"{d}/binarized_gene_expression__stable_genes_by_mutual_information.txt",
                index=False,
            )

        for d in outdirs:
            stable_go.to_csv(
                f"{d}/binarized_gene_expression__stable_genes_by_mutual_information__go_enrichment.csv"
            )

        for d in outdirs:
            unstable_go.to_csv(
                f"{d}/binarized_gene_expression__unstable_genes_by_mutual_information__go_enrichment.csv"
            )

        unstable_genes_series = pd.Series(unstable_gene_names)
        unstable_genes_series = unstable_genes_series.sort_values()
        for d in outdirs:
            unstable_genes_series.to_csv(
                f"{d}/binarized_gene_expression__unstable_genes_by_mutual_information.txt",
                index=False,
            )
        for d in outdirs:
            compartment_group_fractions_all_species[unstable_gene_names].to_csv(
                f"{d}/binarized_gene_expression__unstable_genes_by_mutual_information__fraction_per_compartment.csv"
            )


class BinarizedPrimateAnalyses(BinarizedAnalysesBase):
    """Analyze patterns of on/off genes across species

    Specifically looking for genes with Primate (human-lemur) specificity
    """

    def __init__(
        self,
        adata,
        groupby,
        species_col,
        stable_genes,
        min_fraction=0.1,
        debug=False,
        primates=["Human", "Mouse lemur"],
        species_order=SPECIES_ORDER_HUMAN_FIRST,
        separator="--",
    ):
        super().__init__(
            adata,
            groupby,
            species_col,
            gene_subset=stable_genes,
            debug=debug,
            min_fraction=min_fraction,
            species_order=species_order,
        )
        # This is set after the super init so the group fractions are already filtered
        # for non-expressed genes
        self.gene_lists["background"] = list(self.binarized.columns)
        self.n_genes = len(self.gene_lists["background"])
        self._make_species_adatas()

        self.separator = separator
        self.all_species = self.separator.join(sorted(self.species_order))
        self.group_enrichments = {}
        self.primates = primates
        self.primates_string = self.separator.join(sorted(primates))

    def dotplot(self, gene, groupby=None, standard_scale="var", **kwargs):
        if groupby is None:
            groupby = self.groupby

        for species, ad in self.mini_adatas.items():
            print(f"--- {species} ---")
            sc.pl.dotplot(
                ad,
                gene,
                groupby=groupby,
                expression_cutoff=self.min_fraction,
                standard_scale=standard_scale,
                **kwargs,
            )

    def find_primate_enriched_genes(self):
        # The "pipeline" of finding primate enriched genes
        # TODO: Pycharm tells me it's not good practice to put all these attribute
        #  assignments in a function. So what's a better way to do it?
        self.species_sharing = self._get_species_sharing_per_binarized_group()
        self.species_sharing_counts = self._count_species_sharing()
        logger.debug(
            f"---self.species_sharing_counts --\n{self.species_sharing_counts}"
        )
        self.species_sharing_2d = self._make_species_sharing_2d()
        self.conserved_all = self._get_genes_conserved_in_all()
        self.conserved_any = self._get_genes_conserved_in_any()
        self.primate_any = self._get_primate_enriched_genes()
        self.primate_all = self._get_primate_only_genes()
        self.plot_venns()
        self.do_go_enrichment(plot=True, write=True)

    def _make_species_adatas(self):
        """Make an AnnotatedData object per species"""
        self.mini_adatas = {}

        for species, df in self.adata_shared.obs.groupby(self.species):
            mini_adata = self.adata_shared[df.index, :]
            self.mini_adatas[species] = mini_adata

    def _get_species_sharing_per_binarized_group(self):
        unstacked = self.binarized.unstack()
        species_sharing = unstacked.apply(
            lambda x: self.separator.join(sorted(x.index[x]))
        )
        return species_sharing

    def _count_species_sharing(self):
        species_sharing_counts = self.species_sharing.groupby(level=1).apply(
            lambda x: x.value_counts()
        )
        species_sharing_counts = species_sharing_counts.replace("", np.nan)
        species_sharing_counts = species_sharing_counts.dropna()
        return species_sharing_counts

    def _make_species_sharing_2d(self):
        species_sharing_2d = self.species_sharing.unstack()
        species_sharing_2d = species_sharing_2d.replace("", np.nan)
        logger.debug(f"species_sharing_2d.shape: {species_sharing_2d.shape}")
        species_sharing_2d.head()
        return species_sharing_2d

    def _get_genes_conserved_in_all(self):
        all_species = self.species_sharing_2d == self.all_species
        rows = all_species.all(axis=1, skipna=True)
        conserved_across_all = self.species_sharing_2d.loc[rows]
        conserved_across_all = conserved_across_all.sort_index()
        logger.debug(f"conserved_across_all.shape: " f"{conserved_across_all.shape}")
        logger.debug(
            f"-- conserved_across_all.notnull().sum() --\n"
            f"{conserved_across_all.notnull().sum()}"
        )
        self.n_conserved_all = len(conserved_across_all)
        self._log_n_genes_and_percentage(self.n_conserved_all, "conserved_all")

        self.gene_lists["conserved_all"] = list(conserved_across_all.index)

        for d in self.outdirs:
            csv = (
                f"{d}/binarized_expression__stable_genes__conserved_all__"
                f"{self.groupby}.csv"
            )
            conserved_across_all.to_csv(csv)
        return conserved_across_all

    def _get_genes_conserved_in_any(self):
        all_species = self.species_sharing_2d == self.all_species
        rows = all_species.any(axis=1)
        conserved_across_any = self.species_sharing_2d.loc[rows]
        conserved_across_any = conserved_across_any.sort_index()
        logger.debug(f"conserved_across_any.shape: " f"{conserved_across_any.shape}")
        logger.debug(
            f"-- (conserved_across_any == self.all_species).sum() --\n"
            f"{(conserved_across_any == self.all_species).sum()}"
        )
        self.n_conserved_any = len(conserved_across_any)
        self._log_n_genes_and_percentage(self.n_conserved_any, "conserved_any")

        self.gene_lists["conserved_any"] = list(conserved_across_any.index)

        for d in self.outdirs:
            csv = (
                f"{d}/binarized_expression__stable_genes__conserved_any_"
                f"{self.groupby}.csv"
            )
            conserved_across_any.to_csv(csv)
        return conserved_across_any

    def _get_primate_enriched_genes(self):
        rows = (self.species_sharing_2d == self.primates_string).any(axis=1)
        primate_enriched = self.species_sharing_2d.loc[rows]
        primate_enriched = primate_enriched.sort_index()
        logger.debug(f"primate_enriched.shape: " f"{primate_enriched.shape}")
        logger.debug(
            f'-- (primate_enriched == "{self.primates_string}").sum() --\n'
            f'{(primate_enriched == "{}").sum()}'
        )
        self.n_primate_enriched = len(primate_enriched)
        self._log_n_genes_and_percentage(self.n_primate_enriched, "primate_enriched")

        self.gene_lists["primate_any"] = primate_enriched.index
        for d in self.outdirs:
            csv = (
                f"{d}/binarized_expression__stable_genes__primate_enriched"
                f"__{self.groupby}.csv"
            )
            primate_enriched.to_csv(csv)
        return primate_enriched

    def _get_primate_only_genes(self):
        primate = self.species_sharing_2d == self.primates_string
        isnull = self.species_sharing_2d.isnull()
        rows = (primate | isnull).all(axis=1)
        primate_only = self.species_sharing_2d.loc[rows]
        primate_only = primate_only.dropna(how="all")
        primate_only = primate_only.sort_index()
        logger.debug(f"primate_only.shape: {primate_only.shape}")
        self.n_primate_only = len(primate_only)
        self._log_n_genes_and_percentage(self.n_primate_only, "primate_only")

        self.gene_lists["primate-only"] = primate_only.index
        for d in self.outdirs:
            csv = (
                f"{d}/binarized_expression__stable_genes__primate_only"
                f"__{self.groupby}.csv"
            )
            primate_only.to_csv(csv)
            logger.info(f"Wrote:\n{csv}")
        return primate_only

    def get_per_compartment_sharing_counts(self):
        per_compartment_species_sharing_counts = self.species_sharing_2d.apply(
            lambda x: x.value_counts()
        )
        return per_compartment_species_sharing_counts

    def get_per_compartment_sharing_percentages(self):
        per_compartment_species_sharing_percent = (
            100
            * per_compartment_species_sharing_counts.divide(
                per_compartment_species_sharing_counts.sum(axis=1), axis=0
            )
        )

    def heatmap_sharing_percentages(self):
        sns.heatmap(per_compartment_species_sharing_percent, annot=True)

    def do_per_group_enrichment(self, plot=True, groupby_source=False, write=False):
        """Compute per-group (e.g. per-epithelial) GO enrichment"""
        for gene_subset in self.gene_lists:
            if gene_subset == "primate_any":
                df = self.primate_any
                species = self.primates_string
            elif gene_subset == "conserved_any":
                df = self.conserved_any
                species = self.all_species
            else:
                continue

            df_subset = df.groupby(level=0, axis=1).apply(lambda x: x[x == species])

            background = self.gene_lists["background"]

            self.group_enrichments[gene_subset] = {}

            # Iterate over columns, e.g. epithelial within compartment_group
            for group, series in df_subset.iteritems():
                gene_list = list(series.dropna().index)

                group_enrichment = enrich.do_go_enrichment(
                    gene_list, background=background
                )

                self.group_enrichments[gene_subset][group] = group_enrichment
                if not group_enrichment.empty and plot:
                    self._single_plot_enrichment(
                        group_enrichment,
                        f"{gene_subset} {group}",
                        groupby_source=groupby_source,
                    )
                if not group_enrichment.empty and write:
                    group_sanitized = self.sanitize(group)
                    for d in self.outdirs:
                        csv = (
                            f"{d}/binarized_expression__stable_genes__{gene_subset}"
                            f"__{self.groupby}__{group_sanitized}__functional_enrichment.csv"
                        )
                        group_enrichment.to_csv(csv)
                        logger.info(f"Wrote:\n{csv}")

    def do_per_group_primate_generif(self):
        """Compute per-group (e.g. per-epithelial) GO enrichment"""
        df = self.primate_any
        species = self.primates_string

        df_subset = df.groupby(level=0, axis=1).apply(lambda x: x[x == species])

        mygene_dfs = []

        for group, series in df_subset.iteritems():
            logger.info(f"Querying {species} genes for {group}")
            gene_list = list(series.dropna().index)

            mygene_results = MG.querymany(
                gene_list,
                # Only search the gene summaries
                scopes="symbol,name",
                # Fields to return. Available fields:
                # https://docs.mygene.info/en/latest/doc/data.html#available-fields
                # This is the defaults ("symbol,name,taxid,entrezgene,ensemblgene") in
                # the live API (https://mygene.info/v3/api#/query/get_query) plus
                # "summary"
                fields="symbol,name,summary,generif",
                returnall=True,
                # Only search human
                species="human",
            )
            mygene_df = self.make_mygene_generif_df(mygene_results)
            mygene_df[self.groupby] = group
            mygene_dfs.append(mygene_df)

        self.mygene_results = pd.concat(mygene_dfs, ignore_index=True)

        for d in self.outdirs:
            xlsx = (
                f"{d}/binarized_expression__stable_genes__primate_any"
                f"__{self.groupby}__generif.xlsx"
            )
            logger.info(f"Writing {xlsx} ...")
            with pd.ExcelWriter(xlsx) as writer:
                for name, df in self.mygene_results.groupby(self.groupby):
                    name_sanitized = self.sanitize(name)
                    df.to_excel(writer, sheet_name=name_sanitized, index=False)
            logger.info(f"Done!")

    @staticmethod
    def _mygene_assign(df, key, result):
        try:
            df[key] = result[key]
        except KeyError:
            pass
        return df

    def make_mygene_generif_df(self, mygene_results):
        mygene_results_out = mygene_results["out"]

        result_keys = "query", "name", "summary", "symbol"

        dfs = []

        for result in mygene_results_out:
            try:
                df = pd.DataFrame(result["generif"])
                df["pubmed_url"] = df.pubmed.map(
                    lambda x: PUBMED_TEMPLATE.format(pubmed_id=x)
                )
            except KeyError:
                df = pd.DataFrame({"pubmed": [None], "text": [None]})

            for key in result_keys:
                df = self._mygene_assign(df, key, result)
            dfs.append(df)
        generif_df = pd.concat(dfs, ignore_index=True, sort=True)
        print(generif_df.shape)
        generif_df.head()
        return generif_df


# --- Separate functions for computing mutual information so they are pickle-able --- #
def unpack_binarized(group_index):
    frac, binary = group_index
    return frac, binary


def do_per_column_mutual_information(group1, group2):
    frac1, binary1 = unpack_binarized(group1)
    frac2, binary2 = unpack_binarized(group2)

    mi = binary1.apply(lambda x: normalized_mutual_info_score(x, binary2[x.name]))
    mi.name = "mutual_information"
    mi_df = mi.reset_index()
    mi_df = mi_df.rename(columns={"level_0": "gene_name"})
    mi_df["fraction1"] = frac1
    mi_df["fraction2"] = frac2
    return mi_df


# Guilt-by-association computation of p-value of overlap between two boolean columns
def overlap_pvalue(col1, col2, log=False):
    """Compute probability of overlap for two boolean columns"""

    success_both = (col1 & col2).sum()

    total = len(col1)
    if log:
        cdf = hypergeom.logcdf(M=total, k=success_both, n=col1.sum(), N=col2.sum())
    else:
        cdf = hypergeom.cdf(M=total, k=success_both, n=col1.sum(), N=col2.sum())

    return cdf


def overlap_pvalue_sf(col1, col2, log=False):
    assert len(col1) == len(col2)

    success_both = (col1 & col2).sum()

    total = len(col1)
    if log:
        survival_function = hypergeom.logsf
    else:
        survival_function = hypergeom.sf

    sf = survival_function(M=total, k=success_both, n=col1.sum(), N=col2.sum())

    return sf


def categorize_genes(
    df,
    mouse_col="Mouse",
    lemur_col="Mouse lemur",
    human_col="Human",
    fraction_threshold_for_nearly_constitutive=0.8,
):
    if df.columns.nlevels > 1:
        df.columns = df.columns.droplevel()

    lemur = df[lemur_col]
    human = df[human_col]
    mouse = df[mouse_col]

    no_mouse = not mouse.any()
    no_lemur = not lemur.any()
    no_human = not human.any()

    lemur_mouse = df[[mouse_col, lemur_col]]
    mouse_human = df[[mouse_col, human_col]]
    lemur_human = df[[lemur_col, human_col]]

    no_lemur_mouse = not lemur_mouse.any().all()
    no_mouse_human = not mouse_human.any().all()
    no_lemur_human = not lemur_human.any().all()

    lemur_eq_human = (human == lemur).all()
    mouse_eq_lemur = (lemur == mouse).all()

    n_spots_no_expression_for_nearly_constitutive = (
        len(df.index) * fraction_threshold_for_nearly_constitutive
    )

#     import pdb; pdb.set_trace()

    if df.all().all():
        return "Constitutively expressed in all three species"
    if df.all(axis=1).any() and (df.all(axis=1) == df.any(axis=1)).all():
        return "Conserved, celltype-specific in all three species"
    if (
        df.sum().sum()
        >= len(df.index) * len(df.columns)
        - n_spots_no_expression_for_nearly_constitutive
        and not lemur_eq_human
        and not mouse_eq_lemur
    ):
        return "Nearly constitutively expressed in all three species"
    if (~df).all().all():
        return "Not expressed in all three species"
    if no_mouse and lemur_human.all().all():
        return "Constitutively expressed in Primates, lemur=human"
    if no_lemur and mouse_human.all().all():
        return "Constitutively expressed in Human and Mouse"
    if no_human and lemur_mouse.all().all():
        return "Constitutively expressed in Mouse and Mouse Lemur"
    #     if (~mouse).all() and human[lemur].all():
    #         return "Not expressed in mouse, but same celltypes in Primates"
    if human.all() and no_lemur_mouse:
        return "Constitutively expressed in Human only"
    if lemur.all() and no_mouse_human:
        return "Constitutively expressed in Mouse lemur only"
    if mouse.all() and no_lemur_human and lemur_eq_human:
        return "Constitutively expressed in Mouse only, lemur=human"

    if human.any() and no_lemur_mouse:
        return "Human-specific, celltype-specific"
    if lemur.any() and no_mouse_human:
        return "Lemur-specific, celltype-specific"
    if mouse.any() and no_lemur_human:
        return "Mouse-specific, celltype-specific, lemur=human"
    if no_mouse and lemur_eq_human:
        return "Conserved expression in Primates, celltype-specific, lemur=human"
    if no_human and lemur_mouse.any().any():
        return "Not expressed in human, but expressed in mouse and lemur"
    if (
        lemur_mouse.all(axis=1).any()
        and no_human
        and mouse[lemur].all()
        and lemur[mouse].all()
    ):
        return "Lemur and mouse specific, celltype-specific"
    if (
        mouse_human.all(axis=1).any()
        and not human.all()
        and human[mouse].all()
        and mouse[human].all()
        and no_lemur
    ):
        return "Human and mouse specific, celltype-specific"
    if (
        (~(mouse & human)).all()
        and (mouse[mouse] & lemur[mouse]).all()
        and lemur_eq_human
    ):
        return "Celltype-switching, lemur intermediate"
    if (
        (~(mouse & human)).all()
        and (mouse[mouse] & lemur[mouse]).all()
        and not lemur_eq_human
    ):
        return "Celltype-switching, lemur closer to mouse"
    if (~(mouse & human)).all() and not mouse_eq_lemur and lemur_eq_human:
        return "Celltype-switching, mouse!=human, lemur=human"
    if (mouse[mouse] != lemur[mouse]).all() and lemur_eq_human:
        return "Celltype-switching, lemur=human"
    if (
        (mouse[mouse] & human[mouse]).all()
        and (lemur[human] & human[human]).any()
        and human.sum() > mouse.sum()
        and (~lemur[~(human | mouse)]).all()
    ):
        if lemur.all() and human.all():
            return "Expansion to constitutive in primates, lemur=human"
        elif human.all():
            return "Expansion to constitutive in human"
        elif lemur_eq_human:
            return "Expansion to primates, lemur=human"
        elif human[lemur].all() and lemur[human].all():
            return "Expansion to primates, lemur intermediate"
        else:
            return "Expansion to primates"
    if human.sum() > lemur.sum() and lemur.sum() >= mouse.sum() and mouse_eq_lemur:
        return "Expansion in human, lemur=mouse"
    if (
        (mouse[human] & human[human]).all()
        and (lemur[human] & human[human]).all()
        and mouse.sum() > human.sum()
        and (~lemur[~(human | mouse)]).all()
    ):
        if mouse.all() and lemur.all():
            return "Contraction in human from constitutive in mouse and lemur"
        elif mouse.all():
            return "Contraction in primates to constitutive in mouse"
        elif human[lemur].all() and lemur[human].all():
            return "Contraction in primates, lemur=human"
        else:
            return "Contraction in primates"
    else:
        return "Other"


def plot_binarized_gene(gene, binarized_per_gene_compartment, **kwargs):
    data = binarized_per_gene_compartment[gene]
    annot = data.applymap(lambda x: "On" if x else "")
    sns.heatmap(data, cmap="Blues", annot=annot, fmt="s", vmin=0, vmax=1, **kwargs)


def query_binarized_category_and_plot(
    gene, binarized_per_gene_compartment, context="paper", **kwargs
):
    """Categorize the

    gene: str
        Gene name
    binarized_per_gene_compartment: pd.DataFrame
        A (n_compartment, (genes, species)) shaped dataframe where the
        column axis has genes as the first level and species as the second level

    context : str, None
        If none, use default figure size
    """
    x = binarized_per_gene_compartment[gene]
    category = categorize_genes(x)

    if context == "paper":
        fig, ax = plt.subplots(figsize=(2.5, 2))
    else:
        fig, ax = plt.subplots()
    plot_binarized_gene(gene, binarized_per_gene_compartment, **kwargs)
    ax.set(title=f"{category}, gene: {gene}")

    
    
    
def save_gene_categorization(tissue, celltype_group, per_gene_categorization, figure_folder_base):
    per_gene_categorization.index.name = 'gene_name'
    per_gene_categorization.name = 'gene_category'
    df = per_gene_categorization.reset_index()
    df['tissue'] = tissue
    df['celltype_group'] = celltype_group

    csv = os.path.join(
        figure_folder_base, 
        f'per_gene_categorization__{tissue}__{celltype_group}.csv')
    print(f"Saving csv to: {csv}")
    df.to_csv(csv, index=False)
    
    
    
def compute_bitscores(binarized_per_gene):
    binarized_per_gene_expressed = binarized_per_gene.groupby(level=0, axis=1).filter(
    lambda x: x.any().any()
)
    binarized_per_gene_expressed.head()

    narrow_binarized_bitscore = binarized_per_gene_expressed.groupby(level=0, axis=1).apply(
        lambda x: ''.join('1' if n else '0' for i, n in enumerate(x.values.flatten()))
    )
    return narrow_binarized_bitscore
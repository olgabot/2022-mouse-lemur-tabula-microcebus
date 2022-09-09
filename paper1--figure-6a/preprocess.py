import sys

import numpy as np
import pandas as pd
import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

# Local modules
import plot_utils
import plot_constants

pd.options.display.max_rows = 1000

CELLTYPE_COLS = {
    "bladder": "cell_ontology_class",
    "lung": "narrow_group",
    "muscle": "narrow_group",
    "blood": "narrow_group",
}
COMPARTMENT_COLS = {
    "bladder": "compartment_group",
    "lung": "compartment_group",
    "muscle": "compartment_group",
    "blood": "compartment_group",
}


def add_compartment_combined_cols(adata, separator=": "):
    adata.obs["compartment_broad"] = (
        adata.obs["compartment_group"].astype(str)
        + separator
        + adata.obs["broad_group"].astype(str)
    )
    adata.obs["compartment_narrow"] = (
        adata.obs["compartment_group"].astype(str)
        + separator
        + adata.obs["narrow_group"].astype(str)
    )
    adata.obs["compartment_broad_narrow"] = (
        adata.obs["compartment_broad"]
        + separator
        + adata.obs["narrow_group"].astype(str)
    )

    adata.obs["compartment_species"] = (
        adata.obs["compartment_group"].astype(str)
        + separator
        + adata.obs["species"].astype(str)
    )

    adata.obs["compartment_narrow_species"] = (
        adata.obs["compartment_narrow"].astype(str)
        + separator
        + adata.obs["species"].astype(str)
    )
    new_cols = [
        "compartment_broad",
        "compartment_narrow",
        "compartment_broad_narrow",
        "compartment_narrow_species",
    ]
    adata.obs["species"] = pd.Categorical(
        adata.obs["species"], categories=plot_constants.SPECIES_ORDER, ordered=True
    )
    adata.obs["species_batch"] = pd.Categorical(
        adata.obs["species_batch"],
        categories=plot_constants.SPECIES_BATCH_ORDER,
        ordered=True,
    )
    return adata


def filter_genes_cells(adata):
    # --- Data Exploration/plotting --
    n_counts_col = "n_counts"
    # Quality control - calculate QC covariates
    adata.obs[n_counts_col] = adata.X.sum(1)
    adata.obs["log_counts"] = np.log(adata.obs[n_counts_col] + 1)
    adata.obs["sqrt_counts"] = np.sqrt(adata.obs[n_counts_col])
    adata.obs["n_genes"] = (adata.X > 0).sum(1)
    plot_quality_control(adata, n_counts_col)

    # ## filtering cells on counts
    # Filter cells according to identified QC thresholds:
    print("Total number of cells: {:d}".format(adata.n_obs))
    sc.pp.filter_cells(adata, min_counts=1000)
    print("Number of cells after min count filter: {:d}".format(adata.n_obs))
    # sc.pp.filter_cells(adata, max_counts = 40000)
    # print('Number of cells after max count filter: {:d}'.format(adata.n_obs))
    # adata = adata[adata.obs['mt_frac'] < 0.2]
    # print('Number of cells after MT filter: {:d}'.format(adata.n_obs))
    sc.pp.filter_cells(adata, min_genes=100)
    print("Number of cells after gene filter: {:d}".format(adata.n_obs))
    # ## filtering genes on number of cells
    # Filter genes:
    print("Total number of genes: {:d}".format(adata.n_vars))
    # Min 20 cells - filters out 0 count genes
    sc.pp.filter_genes(adata, min_cells=3)
    print("Number of genes after cell filter: {:d}".format(adata.n_vars))

    # Normalize each cell to have 10,000 counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    plot_quality_control(adata, n_counts_col)

    return adata


def filter_min_cells_per_group(
    adata,
    celltype_col="narrow_group",
    species_col="species",
    min_cells_per_celltype=20,
):
    print(
        f"Filtering for celltype column {celltype_col} with at least"
        f" {min_cells_per_celltype} cells per {species_col}"
    )
    groupby = [celltype_col, species_col]
    print("--- Before filtering ---")
    print(f"-- Number of cell types: {adata.obs[celltype_col].nunique()} --")
    print(adata.obs.groupby(groupby).size())

    cells_to_use = (
        adata.obs.groupby(celltype_col).filter(lambda x: x.species.nunique() == 3).index
    )
    adata_same_narrow_group = adata[cells_to_use]

    df = adata_same_narrow_group.obs.groupby(groupby).filter(
        lambda x: len(x) >= min_cells_per_celltype
    )
    df = df.groupby(celltype_col).filter(lambda x: x[species_col].nunique() == 3)
    adata_min_cells = adata_same_narrow_group[df.index]
    adata_min_cells.obs[celltype_col].cat.remove_unused_categories(inplace=True)
    print("--- After filtering ---")
    print(f"-- Number of cell types: {adata_min_cells.obs[celltype_col].nunique()} --")
    print(df.groupby(groupby).size())
    return adata_min_cells


def count_per_species_cell_types(
    adata,
    celltype_col="narrow_group",
    compartment_col="compartment_group",
    species_col="species",
):
    groupby = [species_col, compartment_col, celltype_col]

    counts = adata.obs.groupby(groupby).size()
    counts.name = "n_cells"
    counts = counts.reset_index()
    # Remove cell types that have 0 counts in a compartment
    counts = counts.groupby([compartment_col, celltype_col]).filter(
        lambda x: (x.n_cells > 0).any()
    )
    print(counts.shape)
    return counts


def celltype_barplot(
    cell_type_counts,
    celltype_col="narrow_group",
    compartment_col="compartment_group",
    species_col="species",
    fig_width=1.5,
    fig_height_scale=0.3,
    context="paper",
    style="whitegrid",
):
    sns.set(context=context, style=style)

    for compartment, df in cell_type_counts.groupby(compartment_col):
        if compartment == "nan":
            continue
        print(f"--- compartment: {compartment} ---")
        fig_height = (df[celltype_col].nunique() + 3) * fig_height_scale
        aspect = fig_width / fig_height
        df[celltype_col] = df[celltype_col].cat.remove_unused_categories()
        df[compartment_col] = df[compartment_col].cat.remove_unused_categories()

        # Replace non-present cell types with 0 for consistent plotting
        df_fillna = (
            df.set_index(species_col)
            .groupby([compartment_col, celltype_col], as_index=False, group_keys=False)
            .apply(
                lambda x: x.reindex(plot_constants.SPECIES_ORDER).fillna(
                    {
                        # Replace any empty celltypes per species with 0
                        "n_cells": 0,
                        compartment_col: x[compartment_col].unique()[0],
                        celltype_col: x[celltype_col].unique()[0],
                    }
                )
            )
        )
        df_fillna = df_fillna.reset_index()

        g = sns.catplot(
            data=df_fillna,
            y=celltype_col,
            x="n_cells",
            hue=species_col,
            hue_order=plot_constants.SPECIES_ORDER,
            palette=plot_constants.SPECIES_TO_COLOR,
            height=fig_height,
            aspect=aspect,
            kind="bar",
            col=compartment_col,
            legend=False,
            zorder=-1,
        )
        g.set_titles("{col_name}")
        g.set(xscale="log", xlim=(0, 1e4), ylabel=None)
        for ax in g.axes.flat:
            ax.set(xticks=[1e1, 1e2, 1e3, 1e4])
            ax.grid(axis="x", color="white", linestyle="-", linewidth=1, zorder=1)
            # ax.axvline(0, y)


def plot_shared_cell_types(
    adata, celltype_col, compartment_col, species_col, fig_height_scale=0.3
):
    counts = count_per_species_cell_types(
        adata, celltype_col, compartment_col, species_col
    )
    celltype_barplot(
        counts,
        celltype_col,
        compartment_col,
        species_col,
        fig_height_scale=fig_height_scale,
    )


def plot_quality_control(adata, n_counts_col):
    t1 = sc.pl.violin(
        adata,
        n_counts_col,
        groupby="species",
        size=2,
        log=True,
        cut=0,
    )
    plot_utils.maximize_figure()
    t2 = sc.pl.violin(
        adata,
        "n_genes",
        groupby="species",
        size=2,
        log=True,
        cut=0,
    )
    plot_utils.maximize_figure()
    t2 = sc.pl.violin(
        adata,
        "sqrt_counts",
        groupby="species",
        size=2,
        log=True,
        cut=0,
    )
    plot_utils.maximize_figure()
    # Data quality summary plots
    fig, ax = plt.subplots()
    p1 = sc.pl.scatter(adata, "n_counts", "n_genes", color="species", ax=ax)
    plot_utils.maximize_figure()
    fig, ax = plt.subplots()
    p2 = sc.pl.scatter(
        adata[adata.obs["n_counts"] < 10000],
        "n_counts",
        "n_genes",
        color="species",
        ax=ax,
    )
    plot_utils.maximize_figure()
    # --- Investigating thresholds ---
    # Thresholding decision: counts
    # Plot all genes
    fig, ax = plt.subplots()
    p3 = sns.distplot(adata.obs[n_counts_col], kde=False, ax=ax)
    plot_utils.maximize_figure()
    plt.show()
    # Plot lowly expressed genes
    fig, ax = plt.subplots()
    p4 = sns.distplot(
        adata.obs[n_counts_col][adata.obs[n_counts_col] < 4000],
        kde=False,
        bins=60,
        ax=ax,
    )
    plot_utils.maximize_figure()
    plt.show()
    # Plot highly expressed genes
    fig, ax = plt.subplots()
    p5 = sns.distplot(
        adata.obs[n_counts_col][adata.obs[n_counts_col] > 10000],
        kde=False,
        bins=60,
        ax=ax,
    )
    plot_utils.maximize_figure()
    plt.show()
    # Thresholding decision: genes
    # Plot all cells, gene counts
    fig, ax = plt.subplots()
    p6 = sns.distplot(adata.obs["n_genes"], kde=False, bins=60, ax=ax)
    plot_utils.maximize_figure()
    plt.show()
    # Plot cells with few genes
    fig, ax = plt.subplots()
    p7 = sns.distplot(
        adata.obs["n_genes"][adata.obs["n_genes"] < 1000], kde=False, bins=60, ax=ax
    )
    plot_utils.maximize_figure()
    plt.show()


def dimensionality_reduction(adata):
    # find highly variable genes
    sc.pp.highly_variable_genes(adata, flavor="cell_ranger")
    print(
        "\n",
        "Number of highly variable genes: {:d}".format(
            np.sum(adata.var["highly_variable"])
        ),
    )
    # # Run PCA
    sc.pp.pca(adata, n_comps=50, use_highly_variable=True, svd_solver="arpack")
    sc.pl.pca_variance_ratio(adata)
    plot_utils.maximize_figure()

    sc.pl.pca_variance_ratio(adata, log=True)
    plot_utils.maximize_figure()
    # # Compute nearest neighbors, UMAP
    # Calculate the visualizations
    sc.pp.pca(adata, n_comps=40)
    plot_utils.maximize_figure()

    sc.pp.neighbors(adata)
    plot_utils.maximize_figure()

    sc.tl.umap(adata)
    return adata


def _plot_umap(adata, color, **kwargs):
    try:
        sc.pl.umap(adata, color=color, **kwargs)
        plot_utils.maximize_figure()
    except NotImplementedError:
        print(f"Error plotting with colors={color}, skipping")


def plot_umaps(adata):
    # ## Plot UMAP by species, age, individual
    _plot_umap(adata, color=["species", "species_batch", "age", "individual"])

    # age_groups, age_palette = remove_unused_palette_keys(
    #     adata, "age", plot_constants.AGE_PALETTE
    # )
    # _plot_umap(
    #     adata,
    #     color="age",
    #     palette=age_palette,
    #     groups=age_groups,
    # )
    #
    # # Plot individuals with custom palette
    # individual_groups, individual_palette = remove_unused_palette_keys(
    #     adata, "individual", plot_constants.INDIVIDUAL_PALETTE
    # )
    #
    # _plot_umap(
    #     adata,
    #     color="individual",
    #     palette=individual_palette,
    #     groups=individual_groups,
    # )

    # ## Plot UMAP by organism sex
    _plot_umap(adata, color="sex", palette="Accent")

    # ## Plot UMAP by cell ontology class
    _plot_umap(adata, color=["cell_ontology_class"])
    # ## Plot UMAP by compartment
    _plot_umap(adata, color=["n_genes", "n_counts"])

    _plot_umap(
        adata,
        color=["species"],
        palette=plot_constants.SPECIES_PALETTE,
        groups=plot_constants.SPECIES_ORDER,
    )
    _plot_umap(adata, color=["cell_ontology_class"])

    cols = ["narrow_group", "broad_group", "compartment_group"]

    # Plot all features at once in same plot for reference, even if colors aren't right
    _plot_umap(adata, color=cols)

    for col in cols:
        if col == "compartment_group":
            # Remove unused compartments
            groups, palette = remove_unused_palette_keys(
                adata, col, plot_constants.compartment_to_color
            )
        else:
            palette = None
            groups = None
        _plot_umap(adata, color=col, palette=palette, groups=groups)


def remove_unused_palette_keys(adata, col, palette):
    # Plot ages with custom palette
    present = set(adata.obs[col])
    palette_present = {k: v for k, v in palette.items() if k in present}
    if len(palette_present) > 0:
        return list(palette_present.keys()), list(palette_present.values())
    else:
        # No values present, don't specify a palette
        return None, None


def run_bbknn(adata):
    # ## run bbknn
    sc.external.pp.bbknn(adata, batch_key="species", metric="euclidean")
    # ### Plot UMAP after BBKNN
    sc.tl.umap(adata)


def remove_color_palettes(adata):
    # Delete all existing color palettes because they mess plotting up
    for key in list(adata.uns.keys()):
        if key.endswith("colors"):
            print(f"removing '{key}'")
            del adata.uns[key]


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    figure_folder = sys.argv[3]
    tissue = sys.argv[4]
    species_col = sys.argv[5]
    min_cells_per_celltype = int(sys.argv[6])

    celltype_col = CELLTYPE_COLS[tissue]
    compartment_col = COMPARTMENT_COLS[tissue]

    adata = sc.read_h5ad(input_file)

    remove_color_palettes(adata)

    adata = add_compartment_combined_cols(adata)
    adata = filter_genes_cells(adata)
    try:
        plot_shared_cell_types(adata, celltype_col, compartment_col, species_col)
        adata = filter_min_cells_per_group(
            adata,
            celltype_col=celltype_col,
            species_col=species_col,
            min_cells_per_celltype=min_cells_per_celltype,
        )
        plot_shared_cell_types(adata, celltype_col, compartment_col, species_col)
        adata = dimensionality_reduction(adata)
        # Pre-BBKNN UMAPs
        plot_umaps(adata)
        run_bbknn(adata)
        # Post-BBKNN UMAPs
        plot_umaps(adata)

        # # Write BBKNN + UMAP data to file
        adata.write(output_file)

        plot_utils.save_figures(plot_figures=True, figure_folder=figure_folder)
    finally:
        # Save all figures so far
        print("--- Exception raised, saving figures so far ---")
        plot_utils.save_figures(plot_figures=True, figure_folder=figure_folder)
    sys.stdout.close()


def print_celltypes_for_makefile(adata, col='narrow_group'):
    print(
        " ".join(
            [
                x.replace(" ", "\\\\")
                .replace("(", "-openparen-")
                .replace(")", "-closedparen-")
                for x in sorted(set(adata.obs[col]))
            ]
        )
    )

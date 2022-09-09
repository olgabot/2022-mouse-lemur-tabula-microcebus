import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp
import seaborn as sns

from plot_constants import SPECIES_ORDER, SPECIES_RENAME

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12

from plot_constants import compartment_to_color


print("Matplotlib Backend: {}".format(plt.get_backend()))
plt.close("all")


def maximize_figure():
    figure_manager = plt.get_current_fig_manager()
    # From https://stackoverflow.com/a/51731824/1628971
    figure_manager.full_screen_toggle()

def compute_ttest_1samp(df, col):
    a = df[col]
    statistic, pvalue = ttest_1samp(a, 0)
    series = pd.Series([statistic, pvalue], index=["statistic", "p_value"])
    return series


def add_significance_stars(order, pvalues, level3=1e-3, level2=1e-2, level1=1e-1):
    ylabels = []

    for ylabel in order:
        if pvalues[ylabel] < level3:
            ylabel += " ***"
        elif pvalues[ylabel] < level2:
            ylabel += " **"
        elif pvalues[ylabel] < level1:
            ylabel += " *"
        ylabels.append(ylabel)
    return ylabels


def correlation_difference_title(anchor_species):
    other_species = [x for x in SPECIES_ORDER if x != anchor_species]
    left = "{}".format(other_species[1])
    right = "{}".format(other_species[0])
    title = (
        left + r" $\leftarrow$ " + f"\t({anchor_species})\t" + r"$\rightarrow$ " + right
    )
    return title


def corr_diff_within(
    df,
    anchor_species,
    compartment_narrow,
    corr_column="xi",
    pvalue_level_kws=None,
):
    if pvalue_level_kws is None:
        pvalue_level_kws = dict(level3=1e-3, level2=1e-2, level1=1e-1)
    other_species = [x for x in SPECIES_ORDER if x != anchor_species]
    correlation_df_pivot_vs = df.query("species1 == @anchor_species")
    correlation_df_pivot_vs_other = correlation_df_pivot_vs.query(
        "species2 != @anchor_species"
    )

    pivot_table = correlation_df_pivot_vs_other.pivot_table(
        index=["species1", "cell_ontology_class", "iteration"],
        columns=["species2"],
        values=corr_column,
    )
    corr_diff_species_str = "{}({}, {}) - {}({}, {})".format(
        corr_column,
        anchor_species,
        SPECIES_RENAME[other_species[0]],
        corr_column,
        anchor_species,
        SPECIES_RENAME[other_species[1]],
    )

    # comparison between human and other species as violin plot
    mi = pivot_table.index
    iteration = mi.get_level_values(2)
    iteration_list = list(iteration.values)
    cell_types = mi.get_level_values(1)
    cell_types_list = list(cell_types.values)
    cell_type_modified_list = []
    for index, cell_type in enumerate(cell_types_list):
        for compartment_group, narrow_group in compartment_narrow.items():
            if cell_type in narrow_group:
                compartment = compartment_group
                break
        try:
            cell_type_modified_list.append(compartment + ": " + cell_type)
        except UnboundLocalError:
            raise ValueError(
                f"Cell type {cell_type} not found assigned to a "
                "compartment in config"
            )

    corr_difference_dataframe = pd.DataFrame(
        columns=["compartment_narrow", corr_diff_species_str, "iteration"]
    )
    print(pivot_table.head())
    diff = pivot_table[other_species[0]].sub(pivot_table[other_species[1]], axis=0)
    diff_list = list(diff.values)
    corr_difference_dataframe[corr_diff_species_str] = diff_list
    corr_difference_dataframe["iteration"] = iteration_list
    corr_difference_dataframe["compartment_narrow"] = cell_type_modified_list
    corr_difference_dataframe = corr_difference_dataframe.reset_index()

    assert len(diff_list) == len(iteration_list) == len(cell_type_modified_list)
    x = corr_diff_species_str
    means = (
        corr_difference_dataframe.groupby("compartment_narrow")[x]
        .mean()
        .sort_values(ascending=False)
    )
    correlation_ttests = corr_difference_dataframe.groupby("compartment_narrow").apply(
        compute_ttest_1samp, corr_diff_species_str
    )
    correlation_ttests["p_value_bonferonni"] = correlation_ttests.p_value * len(
        correlation_ttests.index
    )
    plt.figure()
    fig = sns.distplot(correlation_ttests.p_value)
    fig.set(
        xlabel="Pvalue of {}".format(x), ylabel="Frequency", title="Pvalue distplot"
    )
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=SMALL_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    ylabels = add_significance_stars(
        means.index, correlation_ttests["p_value_bonferonni"], **pvalue_level_kws
    )
    title = correlation_difference_title(anchor_species)

    # comparison between human and other species as violin plot
    colors = {
        x: compartment_to_color[x.split(": ")[0].lower()]
        for x in corr_difference_dataframe.compartment_narrow.unique()
    }
    y = "compartment_narrow"
    x = corr_diff_species_str
    figwidth = 3
    figheight = max(0.1 * len(means.index), 2)
    fig, ax = plt.subplots(figsize=(figwidth, figheight))
#     import pdb; pdb.set_trace()
    print('--- means: ---')
    print(corr_difference_dataframe.groupby(y)[x].mean().to_csv())
    print('--- std dev: ---')
    print(corr_difference_dataframe.groupby(y)[x].std().to_csv())
    sns.barplot(
        data=corr_difference_dataframe, x=x, y=y, order=means.index, palette=colors, ax=ax
    )
    xmin, xmax = ax.get_xlim()
    xabsmax = max(abs(xmin), abs(xmax))
    new_xlim = -xabsmax, xabsmax
    ax.set(
        xlim=new_xlim,
        xlabel="$\Delta$ {} correlation".format(corr_column),
        title=title,
        yticklabels=ylabels,
    )
#     maximize_figure()

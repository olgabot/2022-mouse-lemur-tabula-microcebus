from adjustText import adjust_text
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

sns.set_context("paper")


def add_gene_names(
    x,
    y,
    *args,
    genes_to_label=None,
    significance_multiplier=3,
    logfoldchange_threshold=2,
    arrowprops=dict(arrowstyle="-", color="k", lw=0.5),
    constitutive_genes=None,
    **kwargs,
):
    threshold = logfoldchange_threshold
    x_mask = x.abs() > (logfoldchange_threshold * significance_multiplier)
    y_mask = y.abs() > (logfoldchange_threshold * significance_multiplier)
    significance_mask = x_mask | y_mask
    x_masked = x[significance_mask]
    y_masked = y[significance_mask]

    # Set genes_to_label to all genes if it is None
    genes_to_label = set(x.index) if genes_to_label is None else genes_to_label

    # Set constitutive_genes to empty tuple if it is None
    constitutive_genes = tuple() if constitutive_genes is None else constitutive_genes

    texts = []
    xs_plotted = []
    ys_plotted = []
    for x0, y0, gene_name in zip(x, y, x.index):
        not_ribosomal = not gene_name.startswith("RP")
        not_constitutive = not gene_name in constitutive_genes
        yes_label = gene_name in genes_to_label
        is_interesting = not_ribosomal & not_constitutive & yes_label
        if (abs(x0) > threshold or abs(y0) > threshold) and is_interesting:
            #             text = plt.annotate(
            #                 gene_name,
            #                 xy=(x0, y0),
            #                 arrowprops=arrowprops,
            #                 xytext=(np.sign(x0) * 10, np.sign(y0) * 10),
            #                 textcoords='offset points',
            #             )
            xs_plotted.append(x0)
            ys_plotted.append(y0)
            #         else:
            text = plt.text(x0, y0, gene_name, zorder=100)
            texts.append(text)

    sns.scatterplot(x=xs_plotted, y=ys_plotted, linewidth=0.5, color="red", zorder=10)

    # Get the points already plotted to avoid
    ax = plt.gca()

    # Make axis a little bigger for all the text
    #     xmin, xmax, ymin, ymax = ax.axis()
    #     multiplier = 1.5
    #     ax.set(
    #         xlim=(multiplier * xmin, multiplier * xmax),
    #         ylim=(multiplier * ymin, multiplier * ymax),
    #     )

    scatter = ax.collections
    #     adjust_text(
    #         texts,
    #         np.asarray(xs_plotted),
    #         np.asarray(ys_plotted),
    #         add_objects=scatter,
    #         #         adata=False,
    #         #         Increase allowed overlaps
    #         #         precision=1,
    #         #         # Increase number of iterations
    #         #         lim=5000,
    #         # Add arrow
    #         save_steps=True,
    #         arrowprops=arrowprops,
    #         #         adata=False,
    #     )
    ax.axhline(color="k", zorder=-1)
    ax.axvline(color="k", zorder=-1)


def scatterplot_constitutive(*args, constitutive_genes, **kwargs):
    x, y = args
    overlap = x.index.intersection(constitutive_genes)
    x0 = x[overlap]
    y0 = y[overlap]
    sns.scatterplot(x=x0, y=y0, **kwargs)


def scatterplot_significant(
    x_lfc,
    y_lfc,
    x_neglog10pval,
    y_neglog10pval,
    *args,
    neglog10pval_threshold=5,
    logfoldchange_threshold=2,
    color="steelblue",
    **kwargs,
):
    x_mask = (x_lfc.abs() > logfoldchange_threshold) & (
        x_neglog10pval > neglog10pval_threshold
    )
    y_mask = (y_lfc.abs() > logfoldchange_threshold) & (
        y_neglog10pval > neglog10pval_threshold
    )
    significance_mask = x_mask | y_mask
    x_masked = x_lfc[significance_mask]
    y_masked = y_lfc[significance_mask]
    sns.scatterplot(x=x_masked, y=y_masked, color=color, **kwargs)

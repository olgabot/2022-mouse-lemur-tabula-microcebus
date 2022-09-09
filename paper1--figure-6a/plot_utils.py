import math
import os

import matplotlib.pyplot as plt
from tqdm import tqdm

print("Matplotlib Backend: {}".format(plt.get_backend()))


def get_nrow_ncol(n_items):
    n_col = int(math.ceil(math.sqrt(n_items)))
    n_row = 1
    while n_row * n_col < n_items:
        n_row += 1
    return n_row, n_col


def turn_off_empty_axes(axes):
    for ax in axes.flat:
        ax_has_stuff = ax.lines or ax.collections
        if not ax_has_stuff:
            ax.axis("off")


def save_figures(figure_folder, plot_figures=True):
    if not os.path.exists(figure_folder):
        os.makedirs(figure_folder)
    if plot_figures:
        iterator = tqdm(plt.get_fignums(), desc="Saving figures")
        for i in iterator:
            _save_single_figure(figure_folder, i)


def _save_single_figure(figure_folder, i):
    fig = plt.figure(i)
    fig.savefig(os.path.join(figure_folder, "figure%d.png" % i))
    fig.savefig(os.path.join(figure_folder, "figure%d.pdf" % i))


def maximize_figure():
    figure_manager = plt.get_current_fig_manager()
    # From https://stackoverflow.com/a/51731824/1628971
    figure_manager.full_screen_toggle()

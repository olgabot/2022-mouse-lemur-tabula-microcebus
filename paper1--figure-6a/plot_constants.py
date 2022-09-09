from io import StringIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from matplotlib.colors import rgb2hex
from plot_utils import maximize_figure


def hexify(palette):
    return [rgb2hex(x) for x in palette]


SPECIES_ORDER = ["Mouse", "Mouse Lemur", "Human"]
SPECIES_BATCH_ORDER = ["mouse", "lemur", "human"]

N_SPECIES = len(SPECIES_ORDER)

SPECIES_RENAMER = dict(zip(SPECIES_ORDER, SPECIES_BATCH_ORDER))


sc.settings.verbosity = 3
sc.set_figure_params(
    dpi=200, dpi_save=300, transparent=True, vector_friendly=True, frameon=False
)
sc.logging.print_versions()


SPECIES_ORDER = ["Mouse", "Mouse Lemur", "Human"]
SPECIES_BATCH_ORDER = [
    "mouse_tabula_muris_senis",
    "mouse_ce",
    "lemur",
    "human_hlca",
    "human_sapiens",
]
SPECIES_BATCH_ORDER_HUMAN_FIRST = SPECIES_BATCH_ORDER[::-1]

N_SPECIES = len(SPECIES_ORDER)

tab10 = hexify(sns.color_palette("tab10", n_colors=10))
# Mouse: Orange
# Lemur: Green
# Human: Blue
# Primates: Purple
PRIMATE_COLOR = tab10[4]
SPECIES_PALETTE = [tab10[1], tab10[2], tab10[0]]

SPECIES_PALETTE = [rgb2hex(x) for x in SPECIES_PALETTE]
SPECIES_PALETTE_HUMAN_FIRST = SPECIES_PALETTE[::-1]
SPECIES_TO_COLOR = dict(zip(SPECIES_ORDER, SPECIES_PALETTE))
SPECIES_BATCH_TO_COLOR = dict(zip(SPECIES_BATCH_ORDER, SPECIES_PALETTE))

# COlormaps 
SPECIES_BATCH_TO_COLOR_MAP = {"mouse": "Blues", "lemur": "YlOrBr", "human": "Greens"}
SPECIES_TO_COLOR_MAP = {"Mouse": "Blues", "Mouse lemur": "YlOrBr", "Human": "Greens"}

# sns.palplot(SPECIES_PALETTE)

GROUP_COLS = ["narrow_group", "broad_group", "compartment_group"]
NARROW_GROUP = "narrow_group"

SPECIES_RENAME = {"Human": "human", "Mouse Lemur": "lemur", "Mouse": "mouse"}



compartments = [
    "endothelial",
    "epithelial",
    "lymphoid",
    "myeloid",
    "stromal",  # "immune"
]

compartment_color_table_string = """compartment_v1	dendrogram_group_name	dendrogram_group_color_name	dendrogram_group_color_hex
epithelial	epithelial	dark blue	177ffc
endothelial	endothelial	cyan	41e5e1
stromal	stromal	green	39b353
lymphoid	immune	magenta	c152c1
hematopoietic	immune	dark purple	b07be5
megakaryocyte-erythroid	immune	light purple	9f91fa
myeloid	immune	light purple	9f91fa
neural	neural	red	ed1c24
germ	germ	orange	ff8911"""

n_compartments = len(compartments)
df = pd.read_csv(StringIO(compartment_color_table_string), sep="\t", index_col=0)

compartment_colors = list("#" + df.loc[compartments, "dendrogram_group_color_hex"])
sns.palplot(compartment_colors)
compartment_to_color = dict(zip(compartments, compartment_colors))
ax = plt.gca()
ax.set(xticklabels=compartments, xticks=np.arange(n_compartments))
maximize_figure()


tab20c = hexify(sns.color_palette("tab20c", n_colors=20))
tab20 = hexify(sns.color_palette("tab20", n_colors=20))


AGE_PALETTE = {
    # Human (HLCA) - shades of blue, lightest first
    "46y": tab20c[2],
    "51y": tab20c[1],
    "75y": tab20c[0],
    # Lemur - shades of green, lightest first
    "9y": tab20c[10],
    "10y": tab20c[9],
    "12y": tab20c[8],
    # Mouse - shades of orange, darkest = oldest
    "30m": tab20c[4],
    # Set both 18m and 21m as mid blue since they're both older adult mice,
    # but not the super old ones
    "18m": tab20c[5],
    "21m": tab20c[5],
    "3m": tab20c[6],
    "1m": tab20c[7],
    "nan": "black",
    np.nan: "black",
}


INDIVIDUAL_PALETTE = {
    # Human Lung Cell Atlas- shades of blue
    "human_1_hlca1": tab20c[0],
    "human_2_hlca2": tab20c[1],
    "human_3_hlca3": tab20c[2],
    # Tabula Sapiens -- lighter, aqua blue
    "human_4_tsp2": tab20c[18],
    "human_5_tsp1": tab20c[19],
    # Lemur - shades of green
    "lemur_1_Antoine": tab20c[8],
    "lemur_2_Bernard": tab20c[9],
    "lemur_3_Martine": tab20c[10],
    "lemur_4_Stumpy": tab20c[11],
}

# Mouse - shades of orange
tabula_muris_senis_mouse_ids = [
    "mouse_1_1-M-62",
    "mouse_2_1-M-63",
    "mouse_3_3-F-56",
    "mouse_4_3-F-57",
    "mouse_5_3-M-5/6",
    "mouse_6_3-M-7/8",
    "mouse_7_3-M-8",
    "mouse_8_3-M-8/9",
    "mouse_9_3-M-9",
    "mouse_10_18-F-50",
    "mouse_11_18-F-51",
    "mouse_12_18-M-52",
    "mouse_13_18-M-53",
    "mouse_14_21-F-54",
    "mouse_15_21-F-55",
    "mouse_16_24-M-58",
    "mouse_17_24-M-59",
    "mouse_18_24-M-60",
    "mouse_19_24-M-61",
    "mouse_20_30-M-2",
    "mouse_21_30-M-3",
    "mouse_22_30-M-4",
    "mouse_23_30-M-5",
]
tabula_muris_senis_colors = sns.color_palette(
    "Oranges", n_colors=len(tabula_muris_senis_mouse_ids)
)
tabula_muris_senis_palette = dict(
    zip(tabula_muris_senis_mouse_ids, tabula_muris_senis_colors)
)

INDIVIDUAL_PALETTE.update(tabula_muris_senis_palette)

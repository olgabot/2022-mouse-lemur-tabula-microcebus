INPUT_FILE="/home/olga/data_sm/tabula-microcebus/data-objects/cross-species/concatenated__human-lung--lemur-lung--mouse-lung__10x__one2one_orthologs__unified_compartments__bbknn.h5ad"
OUTPUT_FILE="/mnt/ibm_lg/pranathi/tabula-microcebus-results/lung__10x__concatenated__human_lemur_mouse__one2one_orthologs__preprocessed__xi__narrow_group__drop_if_either_zero.parquet"
VAR := Adventitial\\Fibroblast Airway\\Smooth\\Muscle\\and\\Myofibroblast Alveolar\\Epithelial\\Type\\2 Alveolar\\Fibroblast B\\cell CD4+\\T CD8+\\T Capillary\\Aerocyte Capillary Ciliated Lymphatic Macrophage Monocyte Myeloid\\Dendritic Natural\\Killer Natural\\Killer\\T\\cell Pericyte Plasma Plasmacytoid\\Dendritic Proliferating\\Macrophage Proliferating\\NK/T Vein
RANDOM_SEED="2020"
N_ITERATIONS="1000"
N_JOBS="8"

target: $(VAR)

$(VAR):
	python3  pyxi_any_group_within_celltypes.py ${INPUT_FILE} ${OUTPUT_FILE} narrow_group $@ ${RANDOM_SEED} ${N_ITERATIONS} ${N_JOBS}
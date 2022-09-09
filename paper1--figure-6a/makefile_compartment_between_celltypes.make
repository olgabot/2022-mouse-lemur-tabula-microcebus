INPUT_FILE="/home/olga/data_sm/tabula-microcebus/data-objects/cross-species/concatenated__human-lung--lemur-lung--mouse-lung__10x__one2one_orthologs__unified_compartments__bbknn.h5ad"
OUTPUT_FILE="/mnt/ibm_lg/pranathi/tabula-microcebus-results/lung__10x__concatenated__human_lemur_mouse__one2one_orthologs__preprocessed__xi__compartment_group__drop_if_either_zero.parquet"
CELL_TYPE_NAMES = "endothelial epithelial myeloid stromal lymphoid"
VAR := endothelial epithelial myeloid stromal lymphoid
RANDOM_SEED="2020"
N_ITERATIONS="1000"
N_JOBS="8"

target: $(VAR)

$(VAR):
	python3  pyxi_any_group_between_celltypes.py ${INPUT_FILE} ${OUTPUT_FILE} compartment_group $@ $(CELL_TYPE_NAMES) ${RANDOM_SEED} ${N_ITERATIONS} ${N_JOBS}

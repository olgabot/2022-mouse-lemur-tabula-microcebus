import pandas as pd

def get_celltype_converter(
    tissue,
    column,
    xlsx="/home/olga/googledrive/TabulaMicrocebus/data/cross-species/unified_annotations/Cross_species_unified_annotations_Lung_Muscle_Blood.xlsx",
    group_cols=["narrow_group", "broad_group", "compartment_group"]
):
    sheet_name = f"{tissue}_10X"

    conversions = pd.read_excel(xlsx, sheet_name=sheet_name, header=[0, 1, 2], engine='openpyxl')

    species_to_grouping = conversions.set_index(column)[group_cols]
    species_to_grouping.columns = species_to_grouping.columns.droplevel(level=[1, 2])
    species_to_grouping = species_to_grouping.loc[~species_to_grouping.index.duplicated()]
    species_to_grouping = species_to_grouping.loc[species_to_grouping.index.notnull()]
    species_to_grouping['tissue'] = tissue
    return species_to_grouping

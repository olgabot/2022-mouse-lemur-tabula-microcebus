import csv
import os

import numpy as np
import pandas as pd


from infernal import InfernalParser

DIAMOND_BLASTP_COLUMNS = [
    "read_id",
    "subject_id",
    "percent_identity",
    "e_value",
    "bitscore",
    "subject_title",
    "subject_taxid",
    "subject_species",
    "subject_kingdom",
    "subject_superkingdom",
    "subject_phylum",
]

DIAMOND_PATTERN = "\d+.\d(.+)\[[\w ]+\]"


def get_rfam2go_df(
    version="14.3",
    template="ftp://ftp.ebi.ac.uk/pub/databases/Rfam/{version}/rfam2go/rfam2go",
):
    url = template.format(version=version)
    rfam2go_original = pd.read_csv(url, header=None, sep=" > ")

    df1 = (
        rfam2go_original[0]
        .str.split(expand=True)
        .rename(columns={0: "rfam_id_full", 1: "rfam_description"})
    )
    df2 = (
        rfam2go_original[1]
        .str.split(" ; ", expand=True)
        .rename(columns={0: "go_description", 1: "go_id"})
    )

    ## Make a cleaned rfam2go table
    rfam2go = pd.concat([df1, df2], axis=1)
    rfam2go["rfam_id"] = rfam2go.rfam_id_full.str.split(":").str[1]
    rfam2go = rfam2go.set_index("rfam_id")
    return rfam2go


def read_diamond_blastp_output(
    filename, read_is_uniprot=False, sketch_id_in_basename=False
):
    df = pd.read_csv(filename, sep="\t", names=DIAMOND_BLASTP_COLUMNS)

    if sketch_id_in_basename:
        basename = os.path.basename(filename)
        sketch_id = basename.split("__")[1]
        split = sketch_id.split("_")
        alphabet = split[0].split("-")[1]
        ksize = int(split[1].split("-")[1])
        df["alphabet"] = alphabet
        df["ksize"] = ksize

    if read_is_uniprot:
        df["read_uniprot_id"] = df["read_id"].str.extract(
            "read\d+\/(?P<read_uniprot_id>[\w|]+);"
        )
        df["read_uniprot_id_minimal"] = df["read_uniprot_id"].map(
            lambda x: "|".join(x.split("|")[:2]) if isinstance(x, str) else x
        )

        df["subject_id_minimal"] = df["subject_id"].map(
            lambda x: "|".join(x.split("|")[:2])
        )
        df["read_subject_uniprot_match"] = (
            df["read_uniprot_id_minimal"] == blastp_results["subject_id_minimal"]
        )

    #     df['refseq_id'] = df['subject_id']
    df["description_with_status"] = (
        df.subject_title.str.extract(DIAMOND_PATTERN).iloc[:, 0].str.strip()
    )
    df["additional_status"] = df.subject_title.str.extract("([A-Z]+):")
    df["description"] = (
        df["description_with_status"].str.split(": ").str[-1].str.strip()
    )
    df["description_no_isoform"] = df["description"].str.split(" isoform").str[0]
    df["is_uncharacterized"] = df.subject_title.str.contains("uncharacterized")
    return df


# regex from: https://regex101.com/r/KGoLWn/1/
CELL_BARCODE_CHANNEL_PATTERN = "CB:Z:(?P<cell_barcode>[ACGT]+).+RG:Z:(?P<channel>\w+)"


def read_kmer_csv(
    csv, gene_name_tag="GN", cell_barcode_channel_pattern=CELL_BARCODE_CHANNEL_PATTERN
):
    kmers = pd.read_csv(csv)
    kmers["gene_name"] = kmers["read_name"].str.extract(f"{gene_name_tag}:Z:([\w-]+)")
    kmers["read_id"] = kmers["read_name"].str.split("\t").str[0]
    df = kmers.read_name.str.extractall(cell_barcode_channel_pattern)
    df = df.droplevel(-1)
    kmers_channels = pd.concat([kmers, df], axis=1)
    return kmers_channels



def read_infernal_results(filename):
    df = pd.read_csv(
        InfernalParser(filename)
    )

    # Add inverse e value for weighting
    df["e_value_inverse"] = df["e_value"].apply(lambda x: 1 / x if isinstance(x, float) else x)
    return df


from sklearn.utils.extmath import weighted_mode


def infernal_most_likely_sequence(
    df, id_col="query_accession", weight_col="e_value_inverse"
):
    ids = df[id_col]
    weights = df[weight_col]
    vals, scores = weighted_mode(ids, weights)
    val = vals[0]
    score = scores[0]
    series = pd.Series([val, score], index=[id_col, weight_col + "__weighted"])
    return series


# infernal_results_original.groupby('utar').apply(most_likely_sequence)

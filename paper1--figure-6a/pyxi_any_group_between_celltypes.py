#!/usr/bin/python
import sys
import scanpy
import itertools
import time

import pandas as pd

from pyxi.pyxi import Xi
from pathos import multiprocessing

# ARGUMENTS - INPUT_FILE OUTPUT_FILE GROUP_COL CELL_TYPE CELL_TYPE_NAMES
# N_ITERATIONS N_JOBS RANDOM_SEED


def get_correlation_line(group_name2, obs, iteration, perm):
    (species1, cell1), (species2, cell2) = perm[0], perm[1]
    x1 = subset_x.loc[cell1]
    x2 = subset_ys[obs].loc[cell2]
    zero1 = x1 == 0
    zero2 = x2 == 0

    either_zero = zero1 | zero2
    nonzero_in_one = ~either_zero
    x1 = x1[nonzero_in_one]
    x2 = x2[nonzero_in_one]

    return [
        group_name, group_name2, species1, species2,
        iteration, Xi(x1, x2).correlation]


def bootstrap_between_celltypes(iteration):
    startt_bootstrapping = time.time()
    random_cells1 = dict(
        subset_obs.groupby('species').apply(
            lambda x: x.sample(
                random_state=random_seed + iteration)).index.tolist())

    lines = []
    for obs in range(len(subset_obsys)):
        random_cells2 = dict(
            subset_obsys[obs].groupby('species').apply(
                lambda x: x.sample(
                    random_state=random_seed + 1 + iteration)).index.tolist())

        for index, j in enumerate(itertools.product(random_cells1.items(),
                                                    random_cells2.items())):

            startt = time.time()
            line = get_correlation_line(group_names[obs], obs, iteration, j)
            lines.append(line)
            print(
                "time taken for {} and {} is {:.3f} seconds".format(
                    obs, index, time.time() - startt))
    print(
        "time taken for {} whole bootstrap iteration is {:.3f} seconds".format(
            iteration, time.time() - startt_bootstrapping))
    return lines


def correlate_cell_groups():
    correlation_df = pd.DataFrame(
        columns=[
            col + "1", col + "2", 'species1', 'species2', 'iteration', 'xi'])
    chunksize, extra = divmod(n_iterations, n_jobs)
    if extra:
        chunksize += 1
    pool = multiprocessing.Pool(n_jobs)

    lines = list(
        itertools.chain(*(pool.map(
            lambda i: bootstrap_between_celltypes(i), range(n_iterations),
            chunksize=chunksize))))

    pool.close()
    pool.join()

    for line in lines:
        correlation_df = correlation_df.append(
            {col + "1": line[0],
             col + "2": line[1],
             'species1': line[2],
             'species2': line[3],
             'iteration': line[4],
             'xi': line[5]}, ignore_index=True)

    print(correlation_df.shape)
    print(correlation_df.head())
    return correlation_df


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    col = sys.argv[3]
    group_name = sys.argv[4]
    group_name = group_name.replace("\\", " ")
    groups = sys.argv[5]
    groups = [group for group in sys.argv[5].split(" ")]
    random_seed = int(sys.argv[6])
    n_iterations = int(sys.argv[7])
    n_jobs = int(sys.argv[8])
    adata = scanpy.read_h5ad(input_file)
    adata_subset = adata[adata.obs[col] == group_name]
    subset_x = adata_subset.to_df()
    subset_x.dropna(inplace=True)
    subset_ys = []
    subset_obsys = []
    group_names = []
    for cell_type in groups:
        group_names.append(cell_type)
        adata_subset_y = adata[adata.obs[col] == cell_type]
        adata_subset_y_df = adata_subset_y.to_df()
        adata_subset_y_df.dropna(inplace=True)
        subset_ys.append(adata_subset_y_df)
        subset_obsys.append(adata_subset_y.obs)
    print(sys.argv, len(subset_x), subset_x.head())
    subset_obs = adata_subset.obs
    correlation_df = correlate_cell_groups()
    fmt = output_file.split(".")[-1]
    if "/" in group_name:
        group_name = group_name.replace("/", "_")
    correlation_df.to_parquet(output_file.replace(
        "." + fmt, "_{}_between_celltypes.".format(group_name) + fmt))

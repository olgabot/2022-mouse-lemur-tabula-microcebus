#!/usr/bin/python
import os
import sys
import scanpy
import itertools
import time
from functools import partial

import pandas as pd

from xicor.xicor import Xi
from pathos import multiprocessing


# ARGUMENTS - INPUT_FILE OUTPUT_FILE GROUP_COL CELL_TYPE
# N_ITERATIONS N_JOBS RANDOM_SEED


def xi(iteration, perm):
    (species1, cell1), (species2, cell2) = perm[0], perm[1]
    x1 = subset_x.loc[cell1]
    x2 = subset_x.loc[cell2]
    zero1 = x1 == 0
    zero2 = x2 == 0

    nonzero1 = x1 != 0
    nonzero2 = x2 != 0

    # either_zero = zero1 | zero2
    nonzero_both = nonzero1 & nonzero2
    x1 = x1[nonzero_both]
    x2 = x2[nonzero_both]
    return [
        group_name,
        species1,
        cell1,
        species2,
        cell2,
        iteration,
        Xi(x1, x2).correlation,
    ]


def get_correlation_line(iteration):
    startt = time.time()

    # print(f'iteration: {iteration}, n_iterations: {n_iterations}, (2 * n_iterations): {(2 * n_iterations)}')
    # print(f"iteration + n_iterations: {iteration + n_iterations}, iteration + (2 * n_iterations): {iteration + (2 * n_iterations)}")
    d1 = dict(
        [
            # Random cell from species1
            random_cells1[iteration],
            # Random cell from species2
            random_cells1[iteration + n_iterations],
            # Random cell from species3
            random_cells1[iteration + (2 * n_iterations)],
        ]
    )

    d2 = dict(
        [
            # Random cell from species1
            random_cells2[iteration],
            # Random cell from species2
            random_cells2[iteration + n_iterations],
            # Random cell from species3
            random_cells2[iteration + (2 * n_iterations)],
        ]
    )
    lines = []
    func = partial(xi, iteration)
    for line in map(func, itertools.product(d1.items(), d2.items())):
        lines.append(line)
    print(
        "time taken per iteration %d is %.3f seconds"
        % (iteration, time.time() - startt)
    )
    return lines


def bootstrap_within_group():

    chunksize, extra = divmod(n_iterations, n_jobs)
    if extra:
        chunksize += 1
    pool = multiprocessing.Pool(n_jobs)

    lines = list(
        itertools.chain(
            *(
                pool.map(
                    lambda i: get_correlation_line(i),
                    range(n_iterations),
                    chunksize=chunksize,
                )
            )
        )
    )

    pool.close()
    pool.join()
    return lines


def correlate_cell_groups():
    correlation_df = pd.DataFrame(
        columns=[col, "species1", "species2", "iteration", "xi"]
    )
    lines = bootstrap_within_group()
    for line in lines:
        correlation_df = correlation_df.append(
            {
                col: line[0],
                "species1": line[1],
                "cell1": line[2],
                "species2": line[3],
                "cell2": line[4],
                "iteration": line[5],
                "xi": line[6],
            },
            ignore_index=True,
        )
    print(correlation_df.head())
    print(len(correlation_df))
    return correlation_df


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    col = sys.argv[3]
    group_name = sys.argv[4]
    print(f"group_name original: {group_name}")
    group_name = (
        group_name.replace("\\", " ")
        .replace("-openparen-", "(")
        .replace("-closedparen-", ")")
    )
    print(f"group_name cleaned: {group_name}")
    random_seed = int(sys.argv[5])
    n_iterations = int(sys.argv[6])
    n_jobs = int(sys.argv[7])
    print(f"random_seed: {random_seed}, n_iterations: {n_iterations}, n_jobs: {n_jobs}")
    adata = scanpy.read_h5ad(input_file)
    adata_subset = adata[adata.obs.query(f'{col} == "{group_name}"').index]
    print(f"Number of cells in '{group_name}' per species:")
    print(adata_subset.obs.groupby(["species"]).size())

    try:
        assert adata_subset.obs["species"].nunique() == 3
    except AssertionError:
        raise AssertionError(
            "Not all three species reperesented in this celltype --> skipping"
        )

    subset_x = adata_subset.to_df()
    print(f"sys.argv: {sys.argv}")
    print(f"len(subset_x): {len(subset_x)}")
    print("subset_x.head():\n", subset_x.head())
    subset_x.dropna(inplace=True)
    subset_obs = adata_subset.obs

    # import pdb; pdb.set_trace()

    if subset_obs.empty:
        print("Cell lists are empty --> exiting")
        sys.exit()

    random_cells1 = (
        subset_obs.groupby("species")
        .apply(lambda x: x.sample(n_iterations, replace=True, random_state=random_seed))
        .index.tolist()
    )

    random_cells2 = (
        subset_obs.groupby("species")
        .apply(
            lambda x: x.sample(n_iterations, replace=True, random_state=random_seed + 1)
        )
        .index.tolist()
    )
    print("Number of cells in random_cells1", len(random_cells1))
    print("Number of cells in random_cells2", len(random_cells2))
    correlation_df = correlate_cell_groups()
    fmt = output_file.split(".")[-1]
    dirname = os.path.dirname(output_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    group_name = group_name.replace("/", "-slash-").replace(" ", "_").lower()
    correlation_df.to_parquet(
        output_file.replace("." + fmt, "_{}.".format(group_name) + fmt)
    )

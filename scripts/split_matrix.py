import numpy as np
import argparse
from vampire.common.util import load_sparse, save_sparse
from tqdm import tqdm
import os

def split_sparse(mat, row_divs = [], col_divs = []):
    '''
    mat is a sparse matrix
    row_divs is a list of divisions between rows.  N row_divs will produce N+1 rows of sparse matrices
    col_divs is a list of divisions between cols.  N col_divs will produce N+1 cols of sparse matrices

    return a 2-D array of sparse matrices
    '''
    row_divs = [None]+row_divs+[None]
    col_divs = [None]+col_divs+[None]

    mat_of_mats = np.empty((len(row_divs)-1, len(col_divs)-1), dtype = type(mat))
    for i, (rs, re) in tqdm(enumerate(zip(row_divs[:-1], row_divs[1:])), total=len(row_divs)-1):
        for j, (cs, ce) in enumerate(zip(col_divs[:-1], col_divs[1:])):
            mat_of_mats[i, j] = mat[rs:re, cs:ce]

    return mat_of_mats

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=str)
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--serialization-dir", "-s", type=str)
    args = parser.parse_args()
    if not os.path.isdir(args.serialization_dir):
        os.mkdir(args.serialization_dir)
    master = load_sparse(args.npz)
    mats = split_sparse(master, row_divs=list(np.arange(0, master.shape[0], step=master.shape[0] // args.chunks)))
    total_count = 0
    for ix, mat in enumerate(mats[1:]):
        total_count+= mat[0].shape[0]
        save_sparse(mat[0], os.path.join(args.serialization_dir, f"mat.{ix}.npz"))
        print(f"mat.{ix}.npz: {mat[0].shape[0]} items")
    assert total_count == master.shape[0]
    
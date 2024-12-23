import numpy as np
import pandas as pd
import random

def gaussian_kernel(size, sigma):
    """Generate a 2D Gaussian kernel."""
    kernel = np.fromfunction(
        lambda x, y: (1/ (2*np.pi*sigma**2)) * np.exp(-((x-size//2)**2 + (y-size//2)**2) / (2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)


def get_df(df, i, j, window):
    df = df[(df.bin1 >= i) & (df.bin1 < (i+window))]
    df = df[(df.bin2 >= j) & (df.bin2 < (j+window))]
    if len(df) == 0:
        return None
    return df

def get_df_from_mat(mat, step):
    rows, cols, data = mat.row, mat.col, mat.data
    rows = rows + step
    cols = cols + step
    df = pd.DataFrame({'bin1': rows, 'bin2': cols, 'score': data})
    return df

def get_chrom_size(genome):
    if genome == "hg19":
        chrom_sizes = {
            1: 249250621,
            2: 243199373,
            3: 198022430,
            4: 191154276,
            5: 180915260,
            6: 171115067,
            7: 159138663,
            8: 146364022,
            9: 141213431,
            10: 135534747,
            11: 135006516,
            12: 133851895,
            13: 115169878,
            14: 107349540,
            15: 102531392,
            16: 90354753,
            17: 81195210,
            18: 78077248,
            19: 59128983,
            20: 63025520,
            21: 48129895,
            22: 51304566
        }
    elif genome == "hg38":
        chrom_sizes = {
            1: 248956422,
            2: 242193529,
            3: 198295559,
            4: 190214555,
            5: 181538259,
            6: 170805979,
            7: 159345973,
            8: 145138636,
            9: 138394717,
            10: 133797422,
            11: 135086622,
            12: 133275309,
            13: 114364328,
            14: 107043718,
            15: 101991189,
            16: 90338345,
            17: 83257441,
            18: 80373285,
            19: 58617616,
            20: 64444167,
            21: 46709983,
            22: 50818468
        }
    else:
        raise ValueError("Invalid genome. Please choose either hg19 or hg38.")
    return chrom_sizes


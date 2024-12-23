import pyBigWig
import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description="scale all values in given bigwig by constant factor")
parser.add_argument("-i", "--input_bigwig_file", metavar="input", type=str, help="full path to the input BigWig file")
parser.add_argument("-o", "--output_bigwig_file", metavar="output", type=str, help="full path to the output BigWig file")
parser.add_argument("-s", "--scale_factor", metavar="scale", type=float, help="the scale factor to multiply the values by")
parser.add_argument("-p", "--precision", type=int, default=3, help="number of decimal places to round to")
args = parser.parse_args()

input_bw = pyBigWig.open(args.input_bigwig_file)
output_bw = pyBigWig.open(args.output_bigwig_file, "w")

output_bw.addHeader(list(input_bw.chroms().items()))
chroms = input_bw.chroms()

chunk_size = 100000

for chrom in chroms:
    chrom_size = chroms[chrom]
    print(f'{chrom}...')
    for start in tqdm(range(0, chrom_size, chunk_size)):
        end = min(start + chunk_size, chrom_size)
        values = input_bw.values(chrom, start, end)
        values = np.nan_to_num(values) * args.scale_factor
        values = np.round(values, args.precision)
        starts = np.arange(start, end)
        ends = starts + 1
        output_bw.addEntries([chrom]*len(starts), starts.tolist(), ends=ends.tolist(), values=values.tolist())

input_bw.close()
output_bw.close()

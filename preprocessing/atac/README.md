## ATAC-seq Data Processing

### Step 0 (optional): Generating signal p-value bigwig and peak files from BAM
This step requires the following additional tools/packages:
- bedClip (Download from [UCSC's utilities](https://hgdownload.soe.ucsc.edu/downloads.html#utilities_downloads))
- bedGraphToBigWig (Download from [UCSC's utilities](https://hgdownload.soe.ucsc.edu/downloads.html#utilities_downloads))
- [samtools](https://www.htslib.org/download/)
- [macs2](https://pypi.org/project/MACS2/)
- [bedtools](https://bedtools.readthedocs.io/en/latest/content/installation.html)

It also requires the chromosome sizes (for example, [hg38.chrom.sizes from UCSC](https://hgdownload.soe.ucsc.edu/goldenpath/hg38/bigZips/hg38.chrom.sizes))

Once these are installed successfully, run the following command (as an example)
```
python bam_to_bw.py -f ./data/IMR90_atac.bam -o ./data/IMR90 --bedclip_path ./bedClip --bedgraph2bw_path ./bedGraphToBigWig --chrom_sizes ./data/hg38.chrom.sizes
```
This will create `./data/IMR90.pval.signal.bigwig` and `./data/IMR90.peaks.narrowpeak`. 

<br/>

### Step 1: Cross-cell-type normalization of ATAC bigwig
This step requires installation of [deepTools](https://deeptools.readthedocs.io/en/develop/content/installation.html).
Once this is done, proceed with the following steps
1. Download GM12878 ATAC-seq bigwig which will be used as reference and place it in `./data/`
   ```
   wget https://www.encodeproject.org/files/ENCFF667MDI/@@download/ENCFF667MDI.bigWig -O ./data/GM12878.bigWig
   ```
2. Get scaling factors for each bigwig
   ```
   ./normalize_atac.sh ./data/*.bigwig
   ```
   This will generate `normFactor.txt` which will contain a mapping between the bigwig files and their scaling factors.
3. Scale each bigwig file using the obtained scaling factor
   ```
   python ./scale_bigwig.py -i ./data/IMR90.pval.signal.bigwig -o ./data/IMR90_normalized.bw -s <scaling factor>
   ```
   This will output the normalized bigwig file `./data/IMR90_normalized.bw`.

<br/>

### Step 2: Deduplicate ATAC peaks
Remove ATAC-seq peaks that are within 500bp of each other and keep the peak with maximum intensity. We assume that the peak is of [ENCODE narrowPeak format](https://genome.ucsc.edu/FAQ/FAQformat.html#format12).
```
python dedup_bed.py -f ./data/IMR90.peaks.narrowpeak
```
This will create the deduplicated bed file `./data/IMR90_dedup.bed`

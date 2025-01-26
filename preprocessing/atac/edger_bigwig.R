require(data.table)
require(edgeR)

## turn off exponential notation to avoid rounding errors
options(scipen=999) 

## Load count matrix, removing all columns that we are not interested in:
raw.counts <- read.table("scores.tab", header=TRUE)
raw.counts <- raw.counts[,4:ncol(raw.counts)]

## effective normalization factors are the product 
## of TMM factor and library size for each sample:
norm.factor   <- calcNormFactors(object = raw.counts, method = c("TMM"),refColumn="GM12878.bigWig")
lib.size      <- colSums(raw.counts)
final.factor  <- norm.factor * lib.size

## as one typically scales reads "per million" we divide this factor by 10^6
## we also need the reciprocal value when using bamCoverage later, you'll see why,
## see *comment below
perMillion.factor <- (final.factor / 1000000)^-1

## write to disk:
write.table(x = data.frame(Sample     = names(perMillion.factor),
                           NormFactor = perMillion.factor),
            file = "normFactors.txt", sep="\t", quote = FALSE,
            col.names = TRUE, row.names = FALSE)

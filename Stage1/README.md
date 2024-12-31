## Adding new cell line/condition to config 

A template cell line can be found at `./configs/datamodule/dataset/cells/new_cell.yaml`
1. Copy this file to create `./configs/datamodule/dataset/cells/reed.yaml` (as an example).
2. Replace <cell> with the cell line/condition name (for example, Reed) and provide paths to ATAC-seq bigwig and peaks.
3. Add "reed" to `./configs/datamodule/multicell.yaml` among cells@datasets.
4. Add "reed" to `./configs/datamodule/validation/cross-cell.yaml` as the prediction cell line. If one needs to predict corresponding to a subset of chromosomes, these chromosomes can be added as prediction chromosomes in the same file.

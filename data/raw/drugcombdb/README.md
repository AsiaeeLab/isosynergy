# DrugCombDB raw downloads
- Source page: http://drugcombdb.denglab.org/download
- Accessed: 2026-02-01

## Files
- `drugcombs_response.csv` (dose–response matrices, percent viability; huge)
  - sha256/md5: 3f67aab5efc4df3ae60fc95122b6daf5016f6fde40abc1be115a9683ad69bf73
- `drugcombs_scored.csv` (per-matrix summary synergy scores + cell line metadata)
  - sha256/md5: 094e01cd3a468de345bc2637389a027f26fc4dceada9ce7ac9bfe00e8fa9d433

## Notes
- `drugcombs_response.csv` does not include the cell line; we join `BlockID` (response) to `ID` (scored) to obtain `cell_line`.
- We convert `Response` from percent viability to fraction viability (`response = clamp(Response/100, 0, 1)`) and standardize concentrations to `uM` when units are provided.

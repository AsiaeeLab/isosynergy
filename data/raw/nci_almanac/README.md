# NCI-ALMANAC raw downloads

- Source: https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/
- Accessed: 2026-01-26
- Files:
  - `ComboDrugGrowth_Nov2017.csv` (sha256 `b6e1309b189026c76db3564bab9c49595e099c1aaab01478aa3888df1407a24d`)
  - `NCI60_CELLNAME_to_Combo.txt` (sha256 `d7eebb91748999f2a99c0214636d41f00d138856c28b52498f29d97b3da9d44a`)
  - `NCI_IOA_AOA_drugs.tsv` (sha256 `66d776c93b29123131a88bf3eedf165b02d2bd104471ebec17366855b6c4eacd`)

Notes:
- `ComboDrugGrowth_Nov2017.csv` contains percent growth readouts for two-drug matrices (NSC1 × NSC2) and corresponding single-agent rows (NSC2 is `NA` with `CONCINDEX2 = 0`).
- Cell line aliases follow the NCI60 naming in the combo file; `NCI60_CELLNAME_to_Combo.txt` provides a canonical mapping.
- `NCI_IOA_AOA_drugs.tsv` maps NSC identifiers to generic/preferred drug names and basic annotations.

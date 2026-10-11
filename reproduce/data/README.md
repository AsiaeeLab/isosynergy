# Input sources

The pipeline accepts existing public downloads and verifies their full SHA-256 values in stage 00. It does not install dependencies or retrieve other files.

| Directory under `data/raw/` | Files | Source |
| --- | --- | --- |
| `drugcombdb` | `drugcombs_response.csv`, `drugcombs_scored.csv` | [DrugCombDB download page](http://drugcombdb.denglab.org/download), accessed 2026-02-01 |
| `nci_almanac` | `ComboDrugGrowth_Nov2017.csv`, `NCI60_CELLNAME_to_Combo.txt`, `NCI_IOA_AOA_drugs.tsv` | [NCI-ALMANAC public data mirror](https://ftp.mcs.anl.gov/pub/candle/public/benchmarks/Pilot1/combo/), accessed 2026-01-26 |

`clinical_concentrations.csv` contains 24 drug concentration values. Twenty-three come from Liston and Davis (2017), Tables 1–2; dexamethasone comes from the HEMADY prescribing information, Section 12.3. The table gives the original value and unit, dose schedule, molecular weight and its source, concentration in micromolar, and the conversion for each drug. The analysis retains nonzero doses at most three times this concentration. Drug-name matching is specified in the window script.

`control_membership.csv` contains 674 literature-selected units and their source metadata. It contains no SIR scores or p-values. The analysis joins these rows to regenerated screen results, computes ranks within cell lines and drug pairs, and derives the inactive-pair groups from the single-agent measurements. ALMANAC ComboScores and xenograft categories are the source-reported values used for membership, not independent outcomes measured by this pipeline. Class-level evidence is identified in the group labels. The five-drug and anchor panels are generated from raw block metadata with the stated random seed, independently of the response values.

`control_sources.csv` gives, for each literature key, the full citation, DOI, PubMed record, what was checked against the primary record, a verbatim quote that establishes the control, and what the record does not establish.

The literature keys in the membership table refer to these sources:

| Key | Citation | DOI |
| --- | --- | --- |
| ONeil2016 | O'Neil J, Benita Y, Feldman I, Chenard M, Roberts B, Liu Y, Li J, Kral A, Lejnine S, Loboda A, Arthur W, Cristescu R, Haines BB, Winter C, Zhang T, Bloecher A, Shumway SD. An Unbiased Oncology Compound Screen to Identify Novel Combination Strategies. Mol Cancer Ther 2016;15(6):1155-62. | https://doi.org/10.1158/1535-7163.MCT-15-0843 |
| Holbeck2017 | Holbeck SL, Camalier R, Crowell JA, Govindharajulu JP, Hollingshead M, Anderson LW, Polley E, Rubinstein L, Srivastava A, Wilsker D, Collins JM, Doroshow JH. The National Cancer Institute ALMANAC: A Comprehensive Screening Resource for the Detection of Anticancer Drug Pairs with Enhanced Therapeutic Activity. Cancer Res 2017;77(13):3564-3576. | https://doi.org/10.1158/0008-5472.CAN-17-0489 |
| Hai2017 | Hai J, Liu S, Bufe L, Do K, Chen T, Wang X, Ng C, Li S, Tsao MS, Shapiro GI, Wong KK. Synergy of WEE1 and mTOR Inhibition in Mutant KRAS-Driven Lung Cancers. Clin Cancer Res 2017;23(22):6993-7005. | https://doi.org/10.1158/1078-0432.CCR-17-1098 |
| Montano2013 | Montano R, Thompson R, Chung I, Hou H, Khan N, Eastman A. Sensitization of human cancer cells to gemcitabine by the Chk1 inhibitor MK-8776: cell cycle perturbation and impact of administration schedule in vitro and in vivo. BMC Cancer 2013;13:604. | https://doi.org/10.1186/1471-2407-13-604 |
| Montano2012 | Montano R, Chung I, Garner KM, Parry D, Eastman A. Preclinical development of the novel Chk1 inhibitor SCH900776 in combination with DNA-damaging agents and antimetabolites. Mol Cancer Ther 2012;11(2):427-38. | https://doi.org/10.1158/1535-7163.MCT-11-0406 |
| Hirai2010MCT | Hirai H, Sootome H, Nakatsuru Y, Miyama K, Taguchi S, Tsujioka K, Ueno Y, Hatch H, Majumder PK, Pan BS, Kotani H. MK-2206, an allosteric Akt inhibitor, enhances antitumor efficacy by standard chemotherapeutic agents or molecular targeted drugs in vitro and in vivo. Mol Cancer Ther 2010;9(7):1956-67. | https://doi.org/10.1158/1535-7163.MCT-09-1012 |
| Hirai2009 | Hirai H, Iwasawa Y, Okada M, Arai T, Nishibata T, Kobayashi M, Kimura T, Kaneko N, Ohtani J, Yamanaka K, Itadani H, Takahashi-Suzuki I, Fukasawa K, Oki H, Nambu T, Jiang J, Sakai T, Arakawa H, Sakamoto T, Sagara T, Yoshizumi T, Mizuarai S, Kotani H. Small-molecule inhibition of Wee1 kinase by MK-1775 selectively sensitizes p53-deficient tumor cells to DNA-damaging agents. Mol Cancer Ther 2009;8(11):2992-3000. | https://doi.org/10.1158/1535-7163.MCT-09-0463 |
| Hirai2010CBT | Hirai H, Arai T, Okada M, Nishibata T, Kobayashi M, Sakai N, Imagaki K, Ohtani J, Sakai T, Yoshizumi T, Mizuarai S, Iwasawa Y, Kotani H. MK-1775, a small molecule Wee1 inhibitor, enhances anti-tumor efficacy of various DNA-damaging agents, including 5-fluorouracil. Cancer Biol Ther 2010;9(7):514-22. | https://doi.org/10.4161/cbt.9.7.11115 |
| Leijen2016 | Leijen S, van Geel RMJM, Sonke GS, de Jong D, Rosenberg EH, Marchetti S, Pluim D, van Werkhoven E, Rose S, Lee MA, Freshwater T, Beijnen JH, Schellens JHM. Phase II Study of WEE1 Inhibitor AZD1775 Plus Carboplatin in Patients With TP53-Mutated Ovarian Cancer Refractory or Resistant to First-Line Therapy Within 3 Months. J Clin Oncol 2016;34(36):4354-4361. | https://doi.org/10.1200/JCO.2016.67.5942 |
| Daud2015 | Daud et al. Phase I dose-escalation trial of checkpoint kinase 1 inhibitor MK-8776 as monotherapy and in combination with gemcitabine in patients with advanced solid tumors. J Clin Oncol 2015;33(9):1060-6. | https://doi.org/10.1200/JCO.2014.57.5027 |
| Palma2009 | Palma JP, Wang YC, Rodriguez LE, Montgomery D, Ellis PA, Bukofzer G, Niquette A, Liu X, Shi Y, Lasko L, Zhu GD, Penning TD, Giranda VL, Rosenberg SH, Frost DJ, Donawho CK. ABT-888 confers broad in vivo activity in combination with temozolomide in diverse tumors. Clin Cancer Res 2009;15(23):7277-90. | https://doi.org/10.1158/1078-0432.CCR-09-1245 |

The simulation SD curves are also rebuilt from raw O'Neil replicate measurements. `noise_models.py` rounds the first 13 viability-bin centers and pooled/robust SDs to six decimals. The positive-noise family uses nine selected bins, the bin midpoints specified in the script, and the pooled SD of log viability (lower bound 0.01) rounded to three decimals. These rounding rules are part of the simulation configuration. The full-precision SD table is used for the order and confidence-band calculations.

The BIGL comparison filenames begin with `r1` or `r4` for one or four replicate plates, followed by single-agent strength, noise law, and response condition. Binary `.f64` files store responses and doses; `.i32` files store the per-dataset seeds.

Every generated input has the following path to the public downloads or the two curated tables. File paths in the first column are relative to `reproduce/output/`; script paths are relative to `reproduce/stages/`.

| Generated input | Producer | Inputs and construction |
| --- | --- | --- |
| `inputs/drugcomb_matrices.parquet`, `raw_response.parquet` | `01_inputs/matrices.py` | O'Neil and ALMANAC responses from DrugCombDB, joined to scored-block metadata; ordered dose grids with at least three dose levels per drug and viability conversion |
| `inputs/drugcombdb_block_scores.parquet` | `01_inputs/matrices.py` | The four matrix-level scores in the DrugCombDB scored CSV |
| `inputs/noise_bank_oneil.npz`, `sd_by_mean.csv`, `gross_errors.npz` | `01_inputs/matrices.py` | O'Neil four-block replicates in the DrugCombDB responses; within-group residuals, SD by mean viability, and single-replicate deviations |
| `inputs/almanac_orientation_pairs.npz` | `01_inputs/matrices.py` | ALMANAC drug pairs measured in opposite order; align doses and pair the two response matrices |
| `inputs/noise_model_curves.json`, `oneil_noise_summary.json`, `almanac_noise_summary.json` | `01_inputs/noise_models.py`, `noise_summary.py` | The generated residual banks, SD table, orientation differences, and gross-error sample |
| `inputs/oneil_donors.npz` | `01_inputs/donors.py` | O'Neil residual groups; 20 nearest viability donors outside the target cell line from a seeded sample of 600 four-block groups |
| `inputs/panel_membership.csv` | `01_inputs/panels.py` | Drug, cell-line, and block metadata from the generated matrices; response-independent seeded selection |
| `realdata/meta_*.parquet`, `inputs_*.npz` | `02_realdata/prepare.py`, invoked by stage 01 | DrugCombDB responses, generated dose grids, O'Neil groups and ALMANAC orientation pairs; per-study analysis units and noise references |
| `inputs/nci_grid.parquet`, `nci_sample.csv`, `nci_almanac_baseline_scores.parquet` | `01_inputs/nci_baselines.R` | The three NCI raw files; 20,000 eligible experiments selected with R seed 42; Bliss, HSA, Loewe and ZIP calculations |
| `inputs/*_baseline_disagreement.parquet` | `01_inputs/baseline_summary.py` | Generated DrugCombDB and NCI score tables; correlation, sign disagreement and top-set overlap |
| `inputs/bigl_comparison_data/`, `bigl_real_meta.csv` | `01_inputs/bigl_data.py`, `bigl_real.py` | Defined synthetic Hill surfaces and noise models; 300 selected O'Neil replicate groups, their doses, and per-dataset seeds |
| `inputs/bigl_raw/`, `bigl_runtime.csv` | `03_simulations/bigl.R`, `bigl_runtime.py` | Generated comparison datasets; BIGL 1.9.3 fitting, reference calculations, and measured runtimes |
| `inputs/bb_cache/`, `bb_amplitude_grid.csv`, `bb_amplitudes.csv` | `03_simulations/bb_operating.py` | Defined Hill surfaces, additive and positive noise laws, and the O'Neil SD table; Monte Carlo oracle references and amplitude selection |
| `inputs/bb_*.parquet`, `bb_coverage*.csv`, `bb_quantile_check.csv`, `bb_table_*.csv` | `03_simulations/bb_operating.py`, `bb_coverage.py`, `bb_collect.py`, `bb_search_check.py` | Generated oracle references and amplitudes; confidence bands, rejection rates, power and curve searches |
| `inputs/band_order_*.csv*` | `03_simulations/band_order.py` | The same generated SD table and defined noise laws; paired curve comparisons with common draws |
| Control and concentration-window analyses | `02_realdata/controls.py`, `window.py` | Regenerated screen outputs joined to `data/control_membership.csv` or `data/clinical_concentrations.csv` |
| Other simulation inputs | Scripts in `03_simulations/` | Defined grids, Hill curves, synergy patterns and noise laws, or the generated empirical banks listed above; sample sizes and seeds are specified in each script |

Figure and table scripts read only these generated inputs and stage-02/03 outputs. The final stage reads the released table and those stage outputs; its expected values are comparison targets and are never used to generate fitted or simulated data.

# ADMET Property Prediction

**Goal:** Self-teaching project predicting ADMET properties (drug
absorption/distribution/metabolism/excretion/toxicity) from molecular
structure, using RDKit + PyTDC + scikit-learn.

**Repo:** https://github.com/tuhinc5203/admet-property-prediction

## Completed

- **Week 0 — environment setup**
  - `admet` conda env: RDKit, PyTDC, scikit-learn, pandas, matplotlib, jupyter installed and verified
  - Kernel wired into VS Code; `RDKit-test.ipynb` loads/draws aspirin to confirm the install
  - Git repo initialized and pushed to GitHub
- **Week 1 — data + EDA** (`Data_EDA.ipynb`)
  - All 5 ADMET datasets loaded via PyTDC: `Solubility_AqSolDB`, `BBB_Martins`, `hERG`, `CYP3A4_Veith`, `Clearance_Hepatocyte_AZ`
  - **Solubility EDA done:**
    - 9,982 compounds total, no missing data
    - Regression target `Y` is left-skewed with a long tail (mean -2.86, range -13.17 to 2.14)
    - 37 `Drug_ID` duplicates checked — confirmed distinct molecules (0 duplicate SMILES), left as-is
  - **BBB permeability EDA done:**
    - 2,030 compounds total, no missing data
    - Class balance skewed: 77% permeable (`Y=1`) vs 23% not (`Y=0`) — plan to use AUROC/F1/balanced accuracy, not raw accuracy, when modeling
    - 21 duplicate SMILES in train (unlike solubility); 2 groups have conflicting labels (Trimetrexate, Levodopa — same SMILES labeled both 0 and 1 in the source data) — genuine label noise, deferred to modeling time
    - Data leakage across TDC's own splits: 23 molecules in both train/test, 9 in train/valid, 1 in valid/test — deferred to modeling time
  - **hERG blockade EDA done:**
    - 655 compounds total (smallest dataset so far) — 22 rows missing `Drug_ID` across splits, but `Drug` (SMILES) and `Y` always present, so no modeling impact
    - Class balance skewed: 68% blockers (`Y=1`) vs 32% not (`Y=0`)
    - 5 duplicate SMILES groups in train, 2 with conflicting labels — both sertindole-related analogs (`BMCL20031829_19`/`SERTINDOLE19`, `BMCL20031829_22`/`SERTINDOLE22`), same label-noise pattern as BBB, deferred to modeling time
    - Small leakage: 2 molecules in both train/test, 0 elsewhere — deferred to modeling time
  - **CYP3A4 inhibition EDA done:**
    - 12,328 compounds total (largest dataset so far), no missing data
    - Class balance close to even: 59% non-inhibitor (`Y=0`) vs 41% inhibitor (`Y=1`) — most balanced dataset so far
    - Zero duplicate Drug_ID, zero duplicate SMILES, zero train/valid/test leakage — cleanest dataset so far, no decisions needed
    - `Drug_ID` here is a numeric PubChem CID, not a compound name like the other datasets

## Next steps

- Run the same basic-EDA pass (count, class balance, distribution, missing data) for:
  - Clearance (the last of the 5 datasets)
- At modeling time, decide how to handle the recurring label-conflict and train/valid/test leakage findings from BBB and hERG (see Issues encountered)

## Issues encountered

- **`nbconvert` JSON validation error** when converting a cell's type from code to markdown via `NotebookEdit`: the tool left stale `execution_count`/`outputs` fields on the markdown cell, which `nbconvert --execute` rejected as invalid notebook JSON (it still ran, but printed a validation error). Fixed by stripping `execution_count`/`outputs` from all markdown cells via a one-off script. Watch for this again any time a cell's type is changed rather than freshly inserted.

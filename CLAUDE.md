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
  - **Clearance EDA done — messiest dataset in the panel:**
    - 1,213 compounds total (smallest dataset), no missing data
    - Regression target `Y` heavily right-skewed (mean 42.8, median 19.05, range 3.0–150.0)
    - **Assay censoring:** 16% of rows sit at exactly `Y=3.0`, 11% at exactly `Y=150.0` — strong evidence these are `<3`/`>150` assay boundary values, not real point measurements. Flagged as a modeling constraint (consider censored regression or bucketing near the extremes), not fixed now
    - 94/849 duplicate SMILES (11%, by far the most of any dataset) and heavy train/valid/test leakage (30/54/5 overlaps) — noted, deferred to modeling time, but will need the most deliberate cleanup of the 5 datasets

  **All 5 datasets now have basic EDA complete.** Next EDA task per the project plan (Week 1 steps 3–4): plot MW/LogP/other descriptors against each target, and write a plain-language "why this property matters" note per dataset.
- **Week 1 step 3 — descriptor plots** (MolWt, MolLogP, TPSA, NumHDonors, NumHAcceptors vs target, computed via RDKit): placed inside each dataset's own section, regression datasets use scatter plots, classification datasets use box plots grouped by class.
  - **Solubility done:** MolLogP shows the cleanest, strongest negative trend with target (chemically expected — more lipophilic = less water-soluble). MolWt and TPSA show a weaker "upper bound drops as descriptor increases" pattern. NumHDonors/NumHAcceptors show no strong visible trend. See **Outliers to investigate** below for extreme-value findings from this pass.
  - **BBB done:** TPSA shows the cleanest class separation of any descriptor seen so far (non-permeable centered ~110, permeable centered ~40-50, boxes barely overlap) — independently reproduces the known "TPSA < ~90" rule of thumb for BBB penetration. MolLogP also separates well (permeable skews higher). MolWt shows weak separation.
  - **hERG done:** MolLogP is the strongest separator (blockers skew higher, ~2.5-4 vs ~0-2.5) — consistent with known med-chem knowledge that lipophilic, basic-amine compounds are more prone to hERG liability. TPSA separates mildly in the opposite direction from BBB (non-blockers trend higher). MolWt/HBD/HBA show weak signal.
  - **CYP3A4 done:** MolLogP again the clearest (if noisier) separator — inhibitors center ~3.5-4 vs ~2 for non-inhibitors. TPSA shows essentially no separation here, unlike BBB/hERG — different descriptors matter for different properties. MolWt/HBD/HBA weak.
  - **Clearance done:** no descriptor shows a clean trend — the plots are dominated by the assay-censoring pattern found in basic EDA, visible as dense horizontal bands of points pinned at `Y=150` (ceiling) and `Y=3` (floor) across every descriptor.
  - **All 5 datasets now have descriptor plots complete.**
- **`NOTES.md` created** — a running reasoning/glossary document (why each property matters + technical concept definitions), separate from this file. Meant to become README material later.
- **Outliers investigated — largest-fragment salt stripping added to `compute_descriptors`** (see `NOTES.md`'s "Largest-fragment (salt) stripping" for the full explanation). Applied globally (all 5 datasets re-run), fixed:
  - hERG's Clofilium phosphate: MolWt 1112→339, MolLogP 16.6→6.5 — now correctly represents the single active cation, fully resolved.
  - Solubility's inorganic salts: MolWt/MolLogP shrank substantially (e.g. borate salt 588→59 MolWt, -40.9→-3.9 MolLogP; tungsten salt 2968→248 MolWt, -29.1→-2.6 MolLogP) — no longer absurd, though still not chemically "typical" since the largest fragment is still a small inorganic ion, not an organic drug. Expected and acceptable — see below.

## Outliers to investigate — resolved / remaining

- ✅ **Solubility & hERG multi-fragment salt inflation — resolved by largest-fragment stripping** (see above). Remaining residual weirdness in Solubility's inorganic salts (still unusual LogP values post-stripping) is inherent to those compounds not being organic molecules at all — not a bug, just a property of a few non-drug-like entries in the dataset. No further action planned; flagged as acceptable data noise.
- ⏸️ **CYP3A4 — genuinely large real molecules, not salts, left as-is (decided, not a bug):**
  - PubChem CID 4469 (MolLogP -24.4, MolWt 1505): a large polysulfonated anionic dye-like compound.
  - PubChem CID 6604947 (MolWt 1736, MolLogP 10.8): an avermectin-like macrolide natural product.
  - PubChem CID 434172 (MolWt 1298, MolLogP 20.75): a calixarene-type macrocycle.
  - Decision: keep as-is, don't filter. These are legitimate, real molecules — not data artifacts — just far outside typical oral drug-like size range. Documented here for awareness; revisit only if they turn out to cause modeling problems (e.g. as high-leverage points in a linear model).

## Next steps

- Week 1 step 4: short markdown write-up per dataset on why the property matters in drug development — now largely covered by `NOTES.md`'s "Why each property matters" section, just needs folding into the notebook itself if still wanted there
- At modeling time, decide how to handle:
  - Recurring label-conflict and train/valid/test leakage findings (BBB, hERG, Clearance)
  - Clearance's assay-censoring pattern at Y=3.0/150.0
  - Clearance's especially heavy duplication/leakage (see Issues encountered)
  - Whether `largest_fragment()` should also be applied before actual model featurization (not just EDA descriptor plots) — likely yes, same reasoning applies

## Issues encountered

- **`nbconvert` JSON validation error** when converting a cell's type from code to markdown via `NotebookEdit`: the tool left stale `execution_count`/`outputs` fields on the markdown cell, which `nbconvert --execute` rejected as invalid notebook JSON (it still ran, but printed a validation error). Fixed by stripping `execution_count`/`outputs` from all markdown cells via a one-off script. Watch for this again any time a cell's type is changed rather than freshly inserted.

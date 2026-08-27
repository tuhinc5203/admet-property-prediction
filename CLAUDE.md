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
- **Week 1 — data + EDA — COMPLETE** (`Data_EDA.ipynb`)
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
  - **Week 1 deliverable met:** one combined notebook (`Data_EDA.ipynb`) with all 5 datasets loaded, cleaned, and explored (counts, missing data, class balance/distribution, duplicate/leakage checks, descriptor-vs-target plots, salt-stripping cleanup), each with a "why it matters" pointer to `NOTES.md`. Committed and pushed throughout.
- **Week 2 — featurization + baseline models — COMPLETE** (`Featurization_Baseline.ipynb`, per the project plan). Reviewed and approved by the user on 2026-08-27 — sealed off, no longer pending.
  - Reusable `largest_fragment()`, `compute_morgan_fp()` (2048-bit Morgan fingerprint, radius 2), `featurize()` (fingerprint + MolWt/MolLogP/HBD/HBA/rotatable-bonds, combined into one 2053-column matrix), `dedupe_labels()`, `average_duplicate_targets()`, `load_scaffold_split()` (scaffold split + a swappable `dedup_fn`, defaulting to `dedupe_labels`), `evaluate_regression()`, and `evaluate_classification()` all written and working — matches the project plan's Week 2 tip to write the split/eval boilerplate as reusable functions rather than one-off code per property. (First pass at Solubility/BBB used hand-written one-off load/eval cells; refactored onto these functions afterward, verified identical results before/after.)
  - **Solubility done — first property through the full pipeline:** TDC scaffold split (6,987 train / 1,997 test), baseline `RandomForestRegressor(n_estimators=200)`, no tuning.
    - Results: RMSE 1.286, MAE 0.926, R² 0.686
    - Compared against the live TDC leaderboard (tdcommons.ai/benchmark/admet_group/06aqsol) which reports MAE: top entries range ~0.74 (MiniMol) to ~0.83 (Basic ML), mostly GNN-based (Chemprop, AttentiveFP). Our baseline (MAE 0.926) is worse but in the same ballpark, not wildly off — expected for an untuned RF vs. tuned/GNN methods on a single split (leaderboard averages 5 seeded splits).
    - Checked for the Week 1 duplicate/leakage cleanup on this actual scaffold split: zero duplicate SMILES, zero train/valid/test leakage — already clean, no `dedupe_labels()` needed here.
  - **BBB permeability done:** TDC scaffold split (1,421 train / 406 test before cleaning), baseline `RandomForestClassifier(n_estimators=200, class_weight='balanced')` (balanced to counter the 77%/23% class skew found in Week 1 EDA).
    - **Duplicate/label-conflict cleanup applied at modeling time, as Week 1 deferred** — new `dedupe_labels()` function drops any SMILES with conflicting labels entirely, then dedupes exact repeats. Checked directly on this scaffold split (not the random split Week 1 EDA used): zero train/valid/test leakage (scaffold splitting keeps whole scaffold groups together, so that specific Week 1 leakage concern doesn't apply here), but train had 38 duplicate SMILES (7 label-conflicting groups), test had 12 (2 conflicting) — both cleaned before training/eval. Train/test shrank to 1,376/392 rows.
    - Results after cleaning: ROC-AUC 0.917, F1 0.934, balanced accuracy 0.767 (up slightly from 0.904/0.931/0.762 pre-cleaning — removing conflicting-label noise helped, not just made results "more honest")
    - **Root cause of the conflicting labels, checked directly (not E/Z or R/S stereoisomers — verified no molecule pair had differing SMILES, so they're structurally identical, not different isomers):** the same compound entered twice under different names — a synonym, an old code name, or just different capitalization (e.g. `BRL53080`/`loperamide`, `Trimetrexate`/`trimetrexate`, `acetylsalicylate`/`aspirin`) — each copy apparently sourced from a different literature study with its own permeability cutoff/assay, consistent with BBB_Martins being a multi-source literature compilation. This is why dedup was done on the SMILES string, not `Drug_ID`/name — name-based dedup would have missed all of these.
    - Compared against the live TDC leaderboard (tdcommons.ai/benchmark/admet_group/01bbb): SOTA (MapLight) is 0.916 AUROC, 12/25 entries above 0.9 — our cleaned baseline (0.917) is now at/above that SOTA number on this single split.
  - **Process note:** should have run this duplicate/leakage check *before* first training each model, not after — it was already flagged as a carried-over Week 1 decision. Doing it reactively after a follow-up question worked out fine here, but going forward, check each dataset's carried-over Week 1 issues before training on it, not after.
  - **hERG blockade done — first property run through the fully refactored pipeline** (`load_scaffold_split(Tox, 'hERG')` → `featurize()` → `RandomForestClassifier(class_weight='balanced')` → `evaluate_classification()`, 5 short cells total): scaffold split (456 train / 125 test after dedup), class balance 68%/32% (blockers/non).
    - Results: ROC-AUC 0.851, F1 0.910, balanced accuracy 0.739
    - Compared against the live TDC leaderboard (tdcommons.ai/benchmark/admet_group/20herg): SOTA is MapLight+GNN at 0.880; our baseline (0.851) beats half the top 10 (MiniMol 0.846, RDKit2D+MLP 0.841, Chemprop-RDKit 0.840, ADMETrix 0.836, AttentiveFP 0.825) — the strongest relative-to-leaderboard showing of the three properties done so far.
    - Duplicate/label-conflict cleanup (Week 1 flagged 5 groups in train, incl. the sertindole analogs) applied automatically via `load_scaffold_split()`, no separate step needed this time.
  - **CYP3A4 inhibition done** (`load_scaffold_split(ADME, 'CYP3A4_Veith')` → `featurize()` → `RandomForestClassifier(class_weight='balanced')` → `evaluate_classification()` + AUPRC): scaffold split (8,629 train / 2,467 test), class balance 59%/41% (non-inhibitor/inhibitor), the most even of the three classification tasks. Confirmed clean as Week 1 EDA predicted — `load_scaffold_split()`'s dedup step ran but found nothing to remove.
    - Results: ROC-AUC 0.879, F1 0.750, balanced accuracy 0.779, **AUPRC 0.852**
    - The CYP3A4_Veith TDC leaderboard (tdcommons.ai/benchmark/admet_group/12cyp3a4i) reports **AUPRC, not ROC-AUC** — added `average_precision_score` specifically for a fair comparison. Top 10 ranges Chemprop (0.862) to MapLight+GNN SOTA (0.916); our baseline (0.852) sits just below the whole top 10 — similar story to Solubility (trailing the field, not close to it).
    - F1 (0.750) is notably lower here than BBB's/hERG's F1 — likely because this is the most balanced class split modeled so far, so there's no "easy majority class" propping F1 up the way there was for the skewed datasets.
  - **Clearance done — last of the 5 properties, messiest dataset in the panel:** scaffold split (713 train / 203 test after averaging). Discovered `dedupe_labels()` was wrong for this dataset before applying it — checked first, per the "verify before assuming" habit from BBB — and wrote `average_duplicate_targets()` instead (averages `Y` across duplicate SMILES rather than dropping conflicts; see NOTES.md for the full reasoning). `load_scaffold_split()` now takes a `dedup_fn` parameter so both cleanup strategies share the same loader.
    - Results: RMSE 44.480, MAE 35.069, R² 0.115, **Spearman 0.362**
    - R² of 0.115 looks alarming in isolation, but the TDC leaderboard for this dataset (tdcommons.ai/benchmark/admet_group/17clhepa) reports **Spearman correlation, not R²** — and even SOTA (CFA) only reaches 0.536, with most of the top 10 clustered around 0.43–0.47 and the bottom of the field down to 0.235. Our Spearman of 0.362 sits below the leaderboard's top 10 but is a genuinely meaningful result given how hard this property is for everyone, not just us.
    - Assay-censoring issue (16%/11% of rows pinned at Y=3.0/150.0, flagged in Week 1 EDA) was **not** fixed here, by design — a plain regressor is scored as if those pinned values were exact, a real known limitation carried forward rather than silently patched.
    - In-notebook comparison: re-ran Clearance with `dedupe_labels` instead of `average_duplicate_targets` — Spearman 0.334 (vs. 0.362) on 577 train molecules (vs. 713) — averaging wins on the metric that matters and keeps more molecules, confirming the choice empirically.
  - **Week 2 core deliverable met: all 5 properties have a working baseline model with recorded metrics, each checked against its actual TDC leaderboard metric** (Solubility: MAE, BBB/hERG: ROC-AUC, CYP3A4: AUPRC, Clearance: Spearman). Full results table with leaderboard comparison consolidated in `NOTES.md`'s "Week 2 Results Summary" section for easy review.
  - Zero train/valid/test leakage confirmed directly on all 5 datasets' scaffold splits (not just assumed to generalize from Solubility/BBB) — scaffold splitting eliminates the Week 1-flagged leakage issue across the board.

## Outliers to investigate — resolved / remaining

- ✅ **Solubility & hERG multi-fragment salt inflation — resolved by largest-fragment stripping** (see above). Remaining residual weirdness in Solubility's inorganic salts (still unusual LogP values post-stripping) is inherent to those compounds not being organic molecules at all — not a bug, just a property of a few non-drug-like entries in the dataset. No further action planned; flagged as acceptable data noise.
- ⏸️ **CYP3A4 — genuinely large real molecules, not salts, left as-is (decided, not a bug):**
  - PubChem CID 4469 (MolLogP -24.4, MolWt 1505): a large polysulfonated anionic dye-like compound.
  - PubChem CID 6604947 (MolWt 1736, MolLogP 10.8): an avermectin-like macrolide natural product.
  - PubChem CID 434172 (MolWt 1298, MolLogP 20.75): a calixarene-type macrocycle.
  - Decision: keep as-is, don't filter. These are legitimate, real molecules — not data artifacts — just far outside typical oral drug-like size range. Documented here for awareness; revisit only if they turn out to cause modeling problems (e.g. as high-leverage points in a linear model).

## Next steps

- **Week 3 — improve models + compare approaches** (per the project plan) — not started yet, begin next session:
  - **User's chosen first improvement: add TPSA back into the feature set.** Week 2's `featurize()` deliberately used the plan's literal descriptor list (MW, LogP, HBD/HBA, rotatable bonds) and left TPSA out, even though TPSA was the single strongest signal found in all of Week 1 EDA (near-clean class separation for BBB: non-permeable ~110, permeable ~40-50). The user reviewed Week 2 in full on 2026-08-27 and specifically wants to try adding TPSA as a 6th descriptor and see whether/where it improves the baselines — likely biggest expected impact on BBB given that Week 1 finding, but worth testing across all 5 properties since Week 1 also found TPSA behaves differently per property (e.g. essentially no separation for CYP3A4).
  - Also per the plan (not yet scoped/started): gradient boosting (XGBoost/LightGBM) alongside the RF baselines, hyperparameter tuning via CV on train only, a feature-importance analysis for at least one property, and documenting failure modes.
  - Clearance's assay-censoring pattern at Y=3.0/150.0 is still unaddressed from Week 2 (a plain regressor treats those pinned boundary values as exact) — worth revisiting here too if there's time.

## Deferred / optional ideas

- **GNN comparison for Solubility (optional, post-panel):** classical ML (RF/GBM on fingerprints) is the deliberate choice for the core project — easier to fully explain in an interview than a GNN, per the project plan's own reasoning. But since most of the TDC leaderboard's top Solubility entries are GNN-based (Chemprop, AttentiveFP, MiniMol), once the classical panel across all 5 properties is done, revisit adding a single GNN (e.g. Chemprop) on Solubility as a benchmarked comparison point — good for personal learning and shows range without making the whole project ride on a model that's harder to defend under interview questioning. Not required for the core deliverable.

## Issues encountered

- **`nbconvert` JSON validation error** when converting a cell's type from code to markdown via `NotebookEdit`: the tool left stale `execution_count`/`outputs` fields on the markdown cell, which `nbconvert --execute` rejected as invalid notebook JSON (it still ran, but printed a validation error). Fixed by stripping `execution_count`/`outputs` from all markdown cells via a one-off script. Watch for this again any time a cell's type is changed rather than freshly inserted.

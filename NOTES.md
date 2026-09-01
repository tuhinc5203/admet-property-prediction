# Project Notes — Reasoning & Definitions

Running reference for *why* things are done the way they are in this project — both the biology/pharmacology behind each property, and the reasoning behind technical choices. Meant to be readable on its own, and to become README material later.

## Why each property matters

### Solubility (AqSolDB)
Predicts aqueous solubility — how well a compound dissolves in water. Affects formulation (can it even be made into a usable drug product?) and bioavailability (a compound that won't dissolve can't be absorbed, no matter how good it is at hitting its target).

**Why the target is logS, not raw solubility:** solubility values span many orders of magnitude across compounds — from around 10⁻¹³ mol/L for a very insoluble compound to several mol/L for a highly soluble one. On a raw linear scale, that distribution would be dominated by a handful of extreme outliers, with almost everything else crushed near zero. Solubility is also fundamentally a *multiplicative* quantity, not an additive one — going from 10⁻⁶ to 10⁻⁵ mol/L is "the same size" change as going from 10⁻³ to 10⁻². Taking the log converts that multiplicative scale into an additive one, so equal steps in the target mean equal steps in "how much more/less soluble" — and makes the distribution far more symmetric and better-behaved for regression (this is why the original ESOL dataset and most solubility QSAR literature also report logS).

### BBB Permeability (BBB_Martins)
Predicts whether a compound can cross the blood-brain barrier. Critical for CNS (central nervous system) drug development — a compound can be a perfect drug everywhere else in the body and still completely fail if it can't reach the brain to act on its target. Also matters in the *opposite* direction: a non-CNS drug that unexpectedly crosses the BBB can cause unwanted neurological side effects.

### hERG Blockade
Predicts whether a compound blocks the hERG cardiac ion channel. hERG blockade is one of the most common reasons compounds fail late in development (or get pulled from market) because it can cause dangerous heart arrhythmias. Screened early and often for exactly this reason — failing late, after significant investment, is far more costly than failing early.

### CYP3A4 Inhibition
Predicts whether a compound inhibits the CYP3A4 metabolic enzyme. CYP3A4 alone metabolizes roughly half of all marketed drugs, so a new compound that inhibits it can cause dangerous drug-drug interactions — it can block the body's ability to clear *other* medications a patient is taking, leading to toxic buildup.

### Clearance (Hepatocyte, AZ)
Predicts hepatocyte clearance — how fast the liver metabolizes and eliminates a compound. Governs dosing: a compound cleared too fast may need impractically frequent dosing to stay effective; one cleared too slowly can accumulate to toxic levels.

## Technical concepts & decisions

### The 5 molecular descriptors
Computed via RDKit's `Descriptors` module, used throughout the descriptor-vs-target plots:

| Descriptor | What it measures | Why it matters for ADMET |
|---|---|---|
| **MolWt** | Molecular weight — sum of atomic masses (g/mol) | Bigger molecules generally struggle more to cross membranes, be absorbed, and stay soluble. Lipinski's Rule of Five flags MW > 500 as a red flag. |
| **MolLogP** | Crippen's estimate of LogP — log₁₀ of the octanol/water partition coefficient. Positive = lipophilic (prefers fat), negative = hydrophilic (prefers water) | Governs the tradeoff between membrane crossing (higher LogP = easier through fatty membranes like the BBB) and water solubility (higher LogP = usually less soluble). |
| **TPSA** | Topological polar surface area — surface area (Å²) from polar atoms (O, N, and their attached H) | Proxy for passive membrane diffusion — lower TPSA generally means easier crossing. This is exactly what separated BBB permeability so cleanly (see below). |
| **NumHDonors** | Count of hydrogen-bond donor groups (O–H, N–H) | More donors = more polar = harder membrane crossing. Lipinski flags > 5. |
| **NumHAcceptors** | Count of hydrogen-bond acceptor atoms (roughly N, O) | Same idea as donors. Lipinski flags > 10. |

MolWt, MolLogP, NumHDonors, and NumHAcceptors are literally the four descriptors behind **Lipinski's Rule of Five** (a classic oral drug-likeness rule of thumb). TPSA is a widely used fifth addition, especially relevant for CNS/BBB work.

### Why EDA uses the `train` split only
Every EDA cell in the notebook (counts, missing data, distributions, descriptor plots) scopes to `train`, not the combined train+valid+test data. Looking at valid/test during EDA would let that data quietly influence decisions (e.g. whether to transform a target, which descriptors look useful) before any model has been built — a mild form of information leakage in spirit, even before modeling starts. Scoping to `train` keeps EDA representative of exactly what the model will actually learn from.

### Scatter plots vs. box plots for descriptor-vs-target
- **Regression targets (Solubility, Clearance):** scatter plot, descriptor on x-axis, target on y-axis — directly shows the shape of the relationship.
- **Classification targets (BBB, hERG, CYP3A4):** box plot of the descriptor, grouped by class (`Y=0` vs `Y=1`) — a scatter against a binary 0/1 value isn't informative (everything just piles up on two horizontal lines). A box plot instead shows whether the *distribution* of a descriptor differs between classes.

**What "visible separation" in a box plot actually means:** each box shows the middle 50% of that group's values (25th–75th percentile) with the median marked inside. If the two classes' boxes barely overlap (like BBB's TPSA: `Y=0` centered ~110, `Y=1` centered ~40–50), it means knowing just that one descriptor value would let you guess the class correctly most of the time — that's real predictive signal a model can learn from. If the boxes overlap heavily (like BBB's MolWt), that descriptor carries little information about the class on its own, no matter how sophisticated the downstream model is.

### The RDKit "not removing hydrogen atom without neighbors" warning
RDKit's SMILES parser normally folds explicit hydrogen atoms into an implicit count on their neighboring atom. But some SMILES in these datasets write salts as multiple disconnected fragments, including a standalone, unbonded ion like `[H+]` (a free proton) or `[H-]` (a hydride) with no neighboring atom to fold into — e.g. `[F-].[F-].[H+].[NH4+]` (ammonium bifluoride). RDKit can't remove that hydrogen, so it leaves it as an explicit atom and prints the warning. It's benign — the molecule still parses and descriptors still compute correctly — but it's a symptom of the same salts/mixtures pattern flagged in `CLAUDE.md`'s "Outliers to investigate" section (see there for specific outlier compounds and counts).

### Largest-fragment ("salt") stripping
Some SMILES in these datasets represent salts written as multiple disconnected fragments joined by `.` — e.g. a drug cation plus a counter-ion (hERG's Clofilium phosphate: three copies of the clofilium cation plus one phosphate anion), or an inorganic compound with several ion copies (Solubility's borate/tungsten salts). Computing descriptors on the *full* multi-fragment SMILES sums contributions across every fragment, wildly inflating values like MolWt and MolLogP relative to the actual single molecule.

The fix (`largest_fragment()` in `compute_descriptors`, added to the Solubility section and used by all 5 datasets): after parsing a SMILES, split it into its disconnected fragments (`Chem.GetMolFrags`) and keep only the one with the most heavy atoms — the largest fragment is almost always the actual drug, not a counter-ion. This isn't a guess; it's standard practice in cheminformatics preprocessing (RDKit's own `SaltRemover` utility does something similar with a curated salt list).

**What it fixed and what it didn't:** for Clofilium phosphate, stripping correctly isolated the single active cation (MolWt 1112→339, MolLogP 16.6→6.5 — both now sensible single-molecule values). For Solubility's inorganic salts (a borate, a tungsten oxoanion), stripping shrank the wildly inflated values a lot (MolWt 588→59, 2968→248) but the *fragment itself* is still a small inorganic ion, not an organic drug — so its LogP is still not a chemically typical value. Salt stripping fixes multi-copy inflation; it doesn't turn an inorganic salt into a drug-like molecule. See `CLAUDE.md`'s "Outliers to investigate" for the full before/after numbers.

### Histogram bin count (`bins=50`)
Used for the regression target-distribution histograms (Solubility, Clearance). With datasets in the thousands of rows, too few bins (e.g. the default 10) over-smooths and hides real shape detail (skew, gaps); too many (e.g. 500) makes the plot noisy from sample-size artifacts alone. 50 bins is a reasonable middle ground for datasets this size.

## Featurization & Baseline Modeling

### The model's process: what goes in, what it's compared against
Two separate things are easy to conflate — the *input* the model uses to predict, and the *ground truth* its predictions get checked against. Neither is the same as the model's own prediction.

**What goes in (the features, `X`) — 2053 numbers per molecule, identical set for every property:**

| Piece | Size | What it is |
|---|---|---|
| Morgan fingerprint bits | 2048 columns (`fp_0`...`fp_2047`) | 0/1 flags: does some local substructure (an atom + its neighbors out to 2 bonds) hash to this position? Fine-grained structural detail, not individually human-interpretable. |
| MolWt, MolLogP, NumHDonors, NumHAcceptors, NumRotatableBonds | 5 columns | Lipinski-style descriptors, each with a direct physical meaning — see the descriptor table earlier in this file. |

Same 2053-column recipe for Solubility, BBB, hERG, and CYP3A4 — only `Y` (the target) and the model type (regressor vs. classifier) change between properties, never the feature set. Note `featurize()` uses `NumRotatableBonds` here, not `TPSA` — TPSA was one of Week 1 EDA's 5 descriptors but isn't currently in the Week 2 feature set (the project plan's Week 2 descriptor list swaps it for rotatable bonds). **Decided for Week 3:** add TPSA back in as a 6th descriptor and re-test — chosen specifically because it was the single cleanest class-separator found anywhere in Week 1 EDA (near-clean BBB separation: non-permeable ~110, permeable ~40-50), so it's the most likely single change to move a baseline's score, especially BBB's.

**What it's compared against (the target, `Y`) — real wet-lab measurements, not another prediction:**

| Property | What `Y` actually is | How it was measured |
|---|---|---|
| Solubility | Continuous logS | Real aqueous solubility measurements, aggregated into AqSolDB |
| BBB | Binary permeable/not | Measured brain-to-plasma concentration ratios in animal studies, thresholded |
| hERG | Binary blocker/not | Patch-clamp electrophysiology IC50 measurements against the hERG channel, thresholded |
| CYP3A4 | Binary inhibitor/not | High-throughput enzyme inhibition assay (PubChem bioassay, Veith et al.), thresholded |

**The actual workflow, in order:**
1. **Train:** the model sees (structure → real measurement) pairs for the training molecules only, and learns how structural features relate to the measured outcome.
2. **Predict:** for test molecules — which the scaffold split deliberately makes structurally *different* from anything in training — the model is given only the structure and has to extrapolate a guess, with no access to their real measurements.
3. **Compare:** `evaluate_regression()`/`evaluate_classification()` reveal the test molecules' true `Y` values and score how close the guesses were. Every TDC leaderboard entry is scored the exact same way, against the same `Y` values in the same official test split — that shared ground truth is what makes our numbers and the leaderboard's directly comparable.

Because `Y` itself comes from real (sometimes noisy, sometimes cross-study-inconsistent — see the BBB label-conflict case below) lab measurements rather than a mathematically perfect ground truth, no model should be expected to reach a perfect score — some gap below 1.0 reflects genuine measurement noise in the data, not just model weakness.

### Morgan fingerprints
A model can't read a SMILES string directly — it needs a fixed-length numeric vector. A Morgan fingerprint builds one by walking out from every atom to a fixed `radius` (2 bonds here), hashing each local neighborhood it finds into a position in an `n_bits`-length bit vector (2048 here) and setting that bit to 1. Every molecule, regardless of its actual size, ends up as the same-length vector of structural "this substructure is present somewhere" flags — radius 2 / 2048 bits is the standard cheminformatics default. It's a *bit* vector (present/absent), not a count vector (how many times), which is a reasonable simplification for a baseline model.

### Why fingerprints *and* the 5 descriptors together
Fingerprint bits are structurally rich but not individually interpretable — "bit 1247 is on" means nothing to a chemist. The 5 Lipinski-style descriptors (MolWt, MolLogP, NumHDonors, NumHAcceptors, NumRotatableBonds) are coarser but each has a direct physical meaning. Combining them means the model gets the fine-grained structural detail *and* keeps a human-interpretable subset available for later feature-importance analysis (Week 3) — a pure fingerprint-only model would predict fine but couldn't tell you *why* in terms a chemist recognizes.

### Scaffold split vs. random split
TDC's `get_split(method='scaffold')` groups molecules by their core ring scaffold before splitting, so structurally similar molecules land together in the same split — train and test end up containing genuinely different chemical structures, not just different rows drawn from the same structural neighborhoods. This is a harder, more realistic test of generalization (mirrors the real drug-discovery situation: will this model work on the *next* chemical series, not just interpolate within one it's already seen), and it's how the TDC leaderboard itself evaluates every submission, which is what makes our results directly comparable to it.

One useful side effect discovered while checking Week 1's carried-over leakage concerns: because scaffold splitting keeps whole scaffold groups together, train/valid/test end up with **zero** SMILES overlap by construction — the train/test leakage found in Week 1 EDA (measured on TDC's default *random* split) simply doesn't apply once modeling uses the scaffold split.

### Regression vs. classification metrics
- **Regression (Solubility, Clearance) — RMSE, MAE, R²:** RMSE and MAE are both in the same units as the target; RMSE squares errors before averaging (then square-roots back), so it penalizes a few large misses more heavily than MAE does, which treats every unit of error equally. R² is the fraction of the target's variance the model explains (1.0 = perfect). MAE is the metric the TDC leaderboard itself reports, so it's the number to compare against published results.
- **Classification (BBB, hERG, CYP3A4) — ROC-AUC, F1, balanced accuracy:** chosen back in Week 1 EDA specifically because these datasets have class imbalance (e.g. BBB is 77%/23%), which makes raw accuracy misleading (a model that always predicts the majority class would score high on accuracy while learning nothing). ROC-AUC measures ranking quality across every possible decision threshold, independent of class balance. F1 and balanced accuracy summarize performance at one hard 0/1 threshold — balanced accuracy in particular can reveal a gap ROC-AUC hides (e.g. BBB's ROC-AUC of 0.917 vs. balanced accuracy of only 0.767 — the model ranks well but still gets the hard threshold wrong on the minority class more than the headline number suggests).
- **`class_weight='balanced'`** (used for the classifiers): up-weights the minority class during training so the model isn't just rewarded for defaulting to the majority label.

### Comparing against the live TDC leaderboard
Rather than trusting a metric number in isolation, each baseline gets checked against the actual published TDC leaderboard for that dataset (e.g. `tdcommons.ai/benchmark/admet_group/06aqsol` for Solubility). This gives a concrete sense of where an untuned baseline sits relative to the field — e.g. Solubility's baseline (MAE 0.926) trails the leaderboard's top GNN-based entries (~0.74–0.83) by a real but modest margin, while BBB's baseline (ROC-AUC 0.917) landed at/above the leaderboard's stated SOTA (0.916) on this single split. The leaderboard numbers are usually averaged over 5 different seeded splits, so a single-split baseline number will naturally have more variance than the leaderboard's reported ± range.

### Duplicate SMILES & label-conflict cleanup (`dedupe_labels`)
Distinct from salt-stripping (a chemistry-representation problem, see above), this addresses a data-quality problem: the same molecule (identical SMILES) sometimes appears more than once in a dataset, occasionally with contradictory labels attached. `dedupe_labels()` groups rows by SMILES, drops every row belonging to a SMILES with more than one distinct label outright (there's no way to know which label is correct, so keeping either would be guessing), then removes exact repeats among what's left so a molecule isn't double-counted.

**BBB's specific case, investigated rather than assumed:** the conflicting-label pairs found in BBB's scaffold-split train set were checked for stereoisomerism (E/Z or R/S) as a possible cause — ruled out, since every conflicting pair had byte-for-byte identical SMILES, including the two pairs with chiral (`@`) centers. The actual cause: the same compound entered twice under a synonym, an old development code name, or just different capitalization (`BRL53080` = `loperamide`; `Trimetrexate`/`trimetrexate`; `acetylsalicylate` = `aspirin`) — consistent with BBB_Martins being assembled from multiple literature sources, each possibly using a different permeability cutoff. This is why dedup is done on the **SMILES string**, not `Drug_ID`/name — name-based dedup would have missed all of these, since the names don't look alike.

Applying this to BBB improved results (ROC-AUC 0.904 → 0.917), which is a useful data point on its own: cleaning up label noise isn't just about reporting an "honest" number, it can materially improve what the model learns, because contradictory training examples actively confuse it.

### Duplicate targets for regression (`average_duplicate_targets`) — why it's not `dedupe_labels`
`dedupe_labels()` was built for binary targets, where `nunique(Y) > 1` for a duplicated SMILES genuinely means "contradictory label, drop it." Applying that same logic to a *continuous* target is wrong: two repeat measurements of the same molecule's clearance will almost never land on the exact same float even when the true underlying value is identical, so `nunique() > 1` would be true for nearly every duplicate — checked directly on Clearance's scaffold split, every one of its 136 duplicate-SMILES groups in train (16% of train) had a different `Y` between copies, including gaps as large as 38 vs. 150 for the same molecule. Using `dedupe_labels()` here would have silently discarded 16% of training data as if it were bad data, when it's actually just measurement noise (and, given some of the largest gaps line up with the `Y=3.0`/`Y=150.0` censoring boundaries noted in Week 1 EDA, likely two assay runs where one hit a boundary and one didn't).

`average_duplicate_targets(df, smiles_col='Drug', y_col='Y')` is the regression-appropriate version: group by SMILES, average `Y` across repeats (treating them as noisy readings of one true value), keep the first value for every other column. `load_scaffold_split()` now takes a `dedup_fn` parameter so the same loader serves both cases — `dedupe_labels` (default, for classification) or `average_duplicate_targets` (passed explicitly for regression targets with real duplicates, i.e. Clearance).

Note this doesn't touch the separate assay-censoring issue (values pinned at the `Y=3.0`/`150.0` boundaries) — that's a different, still-unaddressed limitation flagged in Week 1 EDA and carried forward, not something averaging duplicates fixes.

### Choosing the right leaderboard metric per dataset
Worth checking explicitly for every property, not assumed: TDC uses a different metric per dataset depending on what best suits the target's distribution, and it isn't always the "obvious" one.
- Solubility → MAE (not RMSE/R²)
- BBB, hERG → ROC-AUC
- CYP3A4 → **AUPRC** (not ROC-AUC — checked directly on the leaderboard page, easy to assume ROC-AUC by analogy with BBB/hERG and get it wrong)
- Clearance → **Spearman correlation** (not R²/RMSE — rank-order agreement, which matters especially here since assay censoring distorts *exact* values at the extremes but rank order among non-censored molecules is still meaningful)

Each baseline's evaluation cell adds whichever extra metric matches its leaderboard, specifically so the comparison is apples-to-apples rather than eyeballing across mismatched metrics.

## Week 2 Results Summary — Baselines vs. TDC Leaderboard

All 5 properties, one place, for quick review. Every baseline is a single scaffold split + an untuned `RandomForestClassifier`/`Regressor` (`n_estimators=200`, otherwise defaults) — no hyperparameter tuning yet (that's Week 3). Leaderboard scores are each dataset's actual TDC leaderboard metric (checked individually, not assumed by analogy — see above), typically averaged over 5 seeded splits, so they carry less single-split noise than our numbers do.

| Property | Task | Our leaderboard-metric score | Our other metrics | Leaderboard metric | Leaderboard SOTA | Where we land |
|---|---|---|---|---|---|---|
| **Solubility** | Regression | MAE **0.926** | RMSE 1.286, R² 0.686 | MAE | 0.741 (MiniMol) | Below the full top 10 (0.741–0.829) — real but modest gap |
| **BBB** | Classification | ROC-AUC **0.917** | F1 0.934, balanced acc. 0.767 | ROC-AUC | 0.916 (MapLight) | At/above SOTA on this split — best relative showing |
| **hERG** | Classification | ROC-AUC **0.851** | F1 0.910, balanced acc. 0.739 | ROC-AUC | 0.880 (MapLight+GNN) | Mid-pack — beats half the top 10 |
| **CYP3A4** | Classification | AUPRC **0.852** | ROC-AUC 0.879, F1 0.750, balanced acc. 0.779 | AUPRC | 0.916 (MapLight+GNN) | Below the full top 10 |
| **Clearance** | Regression | Spearman **0.362** | RMSE 44.480, MAE 35.069, R² 0.115 | Spearman | 0.536 (CFA) | Below top 10, but this property is hard for everyone (field ranges down to 0.235) |

**Clearance duplicate-handling comparison** (in-notebook, see "Comparison" cells): `average_duplicate_targets` (Spearman 0.362, 713 train molecules) vs. `dedupe_labels` drop-conflicts (Spearman 0.334, 577 train molecules) — averaging wins on the metric that matters and keeps ~16% more distinct molecules, confirming the choice empirically rather than just by argument.

**Overall pattern:** BBB and hERG (the two datasets with the cleanest, most separable descriptor signal back in Week 1 EDA — TPSA and MolLogP respectively) are also where the baseline lands closest to the leaderboard. Solubility, CYP3A4, and Clearance all trail their leaderboards by a more real margin — consistent with most of those leaderboards being dominated by GNN/foundation-model entries (Chemprop, AttentiveFP, MiniMol, MapLight+GNN) rather than classical ML, and with our single-split, untuned baseline being compared against 5-seed-averaged, tuned entries. Closing (some of) that gap is exactly what Week 3's hyperparameter tuning and gradient boosting comparison is for.

## Week 3 — Model Improvement

### Experiment 1: adding TPSA as a 6th descriptor
Week 2's `featurize()` deliberately left TPSA out (the plan's literal Week 2 list was MW/LogP/HBD/HBA/rotatable bonds), despite TPSA being the single cleanest signal found anywhere in Week 1 EDA (near-clean BBB class separation). `Model_Improvement.ipynb` copies Week 2's reusable functions unchanged and modifies only `featurize()` to add TPSA, keeping `Featurization_Baseline.ipynb` sealed as a stable "before" snapshot to compare against (`WEEK2_BASELINE` dict + `compare_to_week2()` in the new notebook).

**Headline result — the hypothesis was wrong in an interesting way.** BBB, the property expected to benefit most, barely moved and its headline metric (ROC-AUC) actually dipped slightly (0.917 → 0.910); Clearance, which had no strong univariate EDA signal for TPSA, got the largest improvement (Spearman 0.362 → 0.391). CYP3A4 was flat, as its Week 1 EDA (no visible separation) predicted. hERG's other metrics dropped, most notably balanced accuracy (0.739 → 0.692).

| Property | Metric | Week 2 → Week 3 (+TPSA) |
|---|---|---|
| Solubility | RMSE / MAE / R² | 1.286→1.280 / 0.926→0.920 / 0.686→0.689 (small improvement) |
| BBB | ROC-AUC | 0.917 → 0.910 (slightly worse) |
| BBB | F1 / balanced acc. | 0.934→0.936 / 0.767→0.779 (slightly better) |
| hERG | ROC-AUC / balanced acc. | 0.851→0.844 / 0.739→0.692 (worse — investigated below) |
| CYP3A4 | ROC-AUC / AUPRC | 0.879→0.878 / 0.852→0.851 (flat) |
| Clearance | Spearman | 0.362 → 0.391 (best improvement of the 5) |

### Investigating the surprise: TPSA's feature-importance rank, and a real RF gotcha
The instinct was "TPSA must be redundant with the fingerprint bits, since a fingerprint can implicitly encode polar-group patterns." Checked directly via `rf.feature_importances_` rather than left as a guess — and the data contradicts that instinct:

| Property | TPSA's importance rank (of 2054 total features) |
|---|---|
| Solubility | 3rd |
| BBB | 2nd |
| hERG | 3rd |
| CYP3A4 | 5th |
| Clearance | **1st** |

TPSA isn't redundant at all — it's near the top of every single model, including CYP3A4 where Week 1 EDA found no visible univariate separation. So why didn't "very important" reliably translate into "measurably better leaderboard score"?

**The actual explanation is a documented statistical artifact, not a coincidence:** scikit-learn's default Random Forest feature importance (mean decrease in impurity, "MDI") is known to be systematically biased toward *continuous* features over *binary* ones. A continuous variable like TPSA offers many possible split thresholds at every tree node; a fixed 0/1 fingerprint bit offers exactly one. More candidate splits means a continuous feature gets selected more often almost mechanically, inflating its MDI importance score independent of how much it actually improves predictions. With 1 continuous TPSA column competing against 2048 binary bits, TPSA was always going to look artificially dominant by this specific metric — "importance rank" and "how much the test score moved" are measuring genuinely different things here, and conflating them would have been a real mistake. (Permutation importance, which measures the actual drop in test performance when a feature is shuffled, is the standard fix for this bias — worth using instead for any feature-importance conclusions the project plan asks for later in Week 3, rather than trusting MDI rank at face value.)

### Investigating hERG's apparent regression: real effect or small-test-set noise?
hERG's test set is only 125 molecules — small enough that a handful of flipped predictions can swing balanced accuracy noticeably. Tested directly rather than assumed: retrained an identical model with the TPSA column dropped from the *already-computed* feature matrix (isolates the effect of that one column, with no other source of variation — same data, same split, same random seed), then compared per-molecule predictions with vs. without TPSA.

**Result: only 3 of 125 test predictions flipped at all**, and every one of them was already a coin-flip call before TPSA was added — predicted probabilities of 0.492, 0.480, and 0.493 (right at the 0.5 decision boundary), nudged by TPSA to 0.505, 0.540, and 0.547. Three borderline flips is entirely sufficient to move balanced accuracy by several points on a 125-molecule test set. **Conclusion: hERG's apparent regression is a small-sample threshold-sensitivity artifact, not TPSA damaging the model.** A larger test set would very likely show this effect shrink toward noise-level.

### Overall takeaway (TPSA experiment)
TPSA carries real, high-ranking signal in every model here — the univariate EDA finding wasn't wrong. But a single added feature's effect on an ensemble model's *aggregate* test score depends on complex interactions with the thousands of features already present, and can be small, mixed, or dataset-specific even when that feature is individually important. The one clear, trustworthy win is Clearance (Spearman +0.029, TPSA ranked #1 there) — everything else is a wash once investigated properly, not a clean "TPSA helped" or "TPSA hurt" story.

### Experiment 2: XGBoost vs. Random Forest (Solubility + BBB)
Random Forest builds trees independently and averages them; gradient boosting (XGBoost) builds trees sequentially, each one specifically targeting the residual error left by the trees before it. Tested on 2 properties rather than the full panel — a comparison to learn from, not a full re-run — reusing the exact same TPSA-augmented features already computed for Experiment 1, so only the model family changes, isolating that as its own variable. Compared against **two** references: Week 2's frozen RF-only baseline, and this notebook's own RF+TPSA result, to separate "does XGBoost help" from "is this just the TPSA effect again."

Same XGBoost result throughout (one run per property) — the two "vs." columns are the same XGBoost number compared against two different starting points, not two different models:

| Property | Metric | Week 2 RF (no TPSA) | Week 3 RF (+TPSA) | XGBoost (+TPSA) | Δ vs. Week 2 RF | Δ vs. Week 3 RF |
|---|---|---|---|---|---|---|
| Solubility | RMSE | 1.286 | 1.280 | **1.264** | −0.022 (better) | −0.016 (better) |
| Solubility | **MAE** (leaderboard metric) | 0.926 | 0.920 | **0.932** | +0.006 (slightly worse) | +0.012 (slightly worse) |
| Solubility | R² | 0.686 | 0.689 | **0.696** | +0.010 (better) | +0.007 (better) |
| BBB | **ROC-AUC** (leaderboard metric) | 0.917 | 0.910 | **0.907** | −0.010 (slightly worse) | −0.003 (essentially flat) |
| BBB | F1 | 0.934 | 0.936 | **0.921** | −0.013 (worse) | −0.015 (worse) |
| BBB | Balanced accuracy | 0.767 | 0.779 | **0.823** | +0.056 (notably better) | +0.044 (notably better) |

**Not a clean win either way — reported honestly rather than spun.** On the metric each dataset's actual TDC leaderboard uses, XGBoost is a wash-to-slightly-worse than the current RF+TPSA baseline (MAE for Solubility, ROC-AUC for BBB both moved the wrong direction by a small amount). But XGBoost clearly does something different and real: Solubility's RMSE/R² both improved (fewer large misses, even though typical-sized error per MAE ticked up slightly — suggesting XGBoost handles outlier cases better but is marginally worse on typical cases), and BBB's balanced accuracy jumped substantially (+0.044 to +0.056) despite ROC-AUC barely moving. That combination (ranking quality flat, hard-threshold accuracy much better) points to `scale_pos_weight` (XGBoost's class-imbalance handling, used here in place of sklearn's `class_weight='balanced'`) producing a better-calibrated decision boundary at the default 0.5 threshold than RF's, without actually improving the model's underlying ability to rank molecules — ROC-AUC measures ranking across every threshold, so it wouldn't reflect this kind of threshold-specific calibration difference.

**Takeaway:** which model "wins" depends on which metric matters for the use case — XGBoost isn't a strict upgrade here, and reporting only one metric (e.g. leading with balanced accuracy for BBB) would have overstated the result.

### Environment issue encountered and fixed: numpy 2.x breaking RDKit
Installing XGBoost via conda pulled `numpy` up to 2.2.6 as a dependency. This RDKit build (2023.09.6) predates numpy 2.0 and is compiled against numpy's older C API — `ConvertToNumpyArray` (used inside `compute_morgan_fp`) failed with `ValueError: Expecting a Numeric array object` as soon as featurization ran again. Fixed by pinning `numpy<2` (conda resolved to 1.26.4), which restored both RDKit's fingerprint conversion and XGBoost working correctly side by side. **Verified the fix didn't silently change anything already committed:** re-ran `Featurization_Baseline.ipynb` (Week 2's sealed notebook) to stdout under the fixed environment and confirmed it reproduces its exact committed numbers (all 5 properties) unchanged. Worth remembering if a future package install pulls numpy back up to 2.x — this specific RDKit build needs numpy 1.x.

### Hyperparameter tuning, explained simply
A quick plain-language summary of what tuning actually changed, kept here for re-explaining later:

- **A hyperparameter is a setting chosen *before* training** — e.g. "how many trees to build," "how deep can each tree grow" — different from what the model *learns from* the data itself (like which fingerprint bits matter). Up to this point, every model just used reasonable-sounding defaults (`n_estimators=200`, etc.). Tuning means actually testing different setting combinations to find ones that work better, instead of guessing.
- **`RandomizedSearchCV` samples instead of trying everything.** The RF settings varied here multiply out to 144 possible combinations — too many to all try. Instead, 15 random combinations were tried and the best-performing one kept. Cheaper than exhaustive search, still finds real improvements.
- **Cross-validation (CV) is how "best" gets judged fairly, without touching the real test set.** For each of the 15 candidate settings, the training data was split into 5 chunks; the model trained on 4 chunks and got checked against the 5th, rotating through all 5 and averaging the result. That way a setting that just got lucky on one split doesn't look artificially good — but the actual held-out test set was never touched during this process, so the final reported numbers are still an honest first look, not something the tuning search already "saw."
- **Project-specific wrinkle: scaffold-grouped CV.** When splitting into those 5 chunks, molecules sharing the same core structure (scaffold) were kept together in the same chunk (via `GroupKFold`), mirroring the same "test on genuinely different molecules" idea behind the original train/test split. Otherwise two near-identical molecules could end up split across chunks, making the tuning look better than it really is.
- **Result:** a tuned XGBoost beat every model tried so far on both properties — and for BBB specifically, it beat the actual published TDC leaderboard score.

### Experiment 3: Hyperparameter tuning (RF + XGBoost, Solubility + BBB)
Both model families tuned via `RandomizedSearchCV` over a small parameter space (15 candidates, not an exhaustive grid — matches the plan's "basic hyperparameter tuning" framing), scored on each dataset's actual leaderboard metric (MAE for Solubility, ROC-AUC for BBB).

**CV uses scaffold-grouped folds (`GroupKFold` on Bemis-Murcko scaffolds via `MurckoScaffold.MurckoScaffoldSmiles`), not plain K-fold.** Plain K-fold would let molecules sharing a scaffold land in different folds — a milder version of the exact leakage scaffold splitting exists to prevent. Grouping keeps every CV fold as realistic a test as the actual train/test split.

`class_weight='balanced'` (RF) / `scale_pos_weight` (XGBoost) were held fixed rather than searched, so tuning targets genuine hyperparameters, not the imbalance-handling choice already made in Week 2/Experiment 2.

**Full results, both properties, all four candidate models plus the two frozen baselines:**

| Property | Metric | Week 2 RF | Week 3 RF+TPSA | Untuned XGB | Tuned RF | **Tuned XGBoost** |
|---|---|---|---|---|---|---|
| Solubility | RMSE | 1.286 | 1.280 | 1.264 | 1.256 | **1.235** |
| Solubility | **MAE** | 0.926 | 0.920 | 0.932 | 0.912 | **0.899** |
| Solubility | R² | 0.686 | 0.689 | 0.696 | 0.700 | **0.710** |
| BBB | **ROC-AUC** | 0.917 | 0.910 | 0.907 | 0.915 | **0.920** |
| BBB | F1 | 0.934 | 0.936 | 0.921 | 0.925 | **0.924** |
| BBB | Balanced accuracy | 0.767 | 0.779 | 0.823 | 0.794 | **0.836** |

**Tuned XGBoost wins on nearly every metric for both properties** — a real, unambiguous result this time, unlike Experiments 1 and 2's mixed outcomes. For BBB, tuned XGBoost's ROC-AUC (0.920) now **exceeds the TDC leaderboard's stated SOTA (0.916)** on this single split. Best hyperparameters found: Solubility XGBoost — `n_estimators=200, max_depth=9, learning_rate=0.05, subsample=1.0, colsample_bytree=0.7`; BBB XGBoost — `n_estimators=400, max_depth=5, learning_rate=0.1, subsample=0.85, colsample_bytree=0.5`.

### Bug encountered and fixed: XGBoost + scikit-learn version mismatch silently producing NaN CV scores
BBB's XGBoost tuning search initially returned `Best CV ROC-AUC: nan` — investigated rather than reported as-is, since a NaN CV score means hyperparameter selection isn't actually comparing candidates meaningfully (whichever candidate sklearn ranks "best" among a set of NaN scores is essentially arbitrary). Confirmed both classes were present in every CV fold (ruled out a missing-class-in-fold explanation), then reproduced the exact failure with `error_score='raise'` to force the real exception instead of a silent NaN:

```
ValueError: XGBClassifier should either be a classifier to be used with
response_method=predict_proba or the response_method should be 'predict'.
Got a regressor with response_method=predict_proba instead.
```

**Root cause:** installing XGBoost pulled scikit-learn up to 1.7.2 as a dependency (see the numpy issue above — same install). Scikit-learn 1.7's newer classifier-auto-detection code (used internally by the `'roc_auc'` string scorer to decide whether to call `predict_proba` or `predict`) doesn't correctly recognize this XGBoost version's `XGBClassifier` as a classifier in that specific code path — even though `.predict_proba()` works completely normally when called directly (which is why `evaluate_classification()` and Experiment 2's untuned XGBoost were unaffected — they call `predict_proba` explicitly, never going through this scorer machinery). `RandomForestClassifier` is unaffected; this is XGBoost-specific.

**Fix:** replaced the `'roc_auc'` string scorer with an explicit callable scorer (`proba_roc_auc`) that calls `estimator.predict_proba(X)[:, 1]` directly, bypassing sklearn's broken auto-detection entirely — no environment changes needed. Verified the fix directly (reran the exact failing case standalone: `nan` → 5 valid fold scores ~0.87–0.93) before trusting the notebook's re-run. Worth remembering: any future `scoring='roc_auc'` (or similar string scorer) usage with this specific XGBoost/scikit-learn combination should use an explicit callable scorer instead.

### Overall pattern (all three experiments)
BBB and hERG (cleanest, most separable descriptor signal in Week 1 EDA) started closest to the leaderboard, and BBB is now the strongest result in the whole panel after tuning — genuinely exceeding leaderboard SOTA, not just "close." Solubility, CYP3A4, and Clearance still trail their leaderboards by a real margin even after tuning (Solubility's tuned XGBoost MAE of 0.899 is closer to the leaderboard's 0.741–0.829 range than the original 0.926 baseline was, but still below it) — consistent with those leaderboards being dominated by GNN/foundation-model entries. Unlike TPSA and untuned-XGBoost's mixed, metric-dependent effects, tuning was the first Week 3 change to produce an unambiguous win.

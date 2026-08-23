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

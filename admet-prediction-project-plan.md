# ADMET Property Prediction Project — Full Build Plan

**Goal:** Build a small panel of machine learning models that predict ADMET properties (Absorption, Distribution, Metabolism, Excretion, Toxicity) directly from a molecule's chemical structure. This is one of the most universally recognized early-stage drug discovery tasks in industry, which is exactly why it's a strong resume project for target companies like Genentech, Amgen, Gilead, and AbbVie.

**Why this project specifically:** Every one of those companies has computational teams that do exactly this — triaging compounds before committing synthesis and lab resources to them. You don't need deep learning or a huge compute budget to demonstrate the core competency; you need clean data handling, correct use of molecular featurization, sound model evaluation, and clear communication of results. This plan is designed to teach you those things in sequence, assuming light-to-no prior coding experience.

**Total estimated time:** 5–6 weeks, at roughly 5–8 hours/week alongside coursework. Not a sprint — this is meant to be steady, part-time work.

---

## Before you start: tools you'll need

| Tool | Purpose | Notes |
|---|---|---|
| VS Code | Your editor/IDE | You already have this |
| Claude Code (VS Code extension) | AI coding assistant, runs on your $20/mo Pro plan | Install from the Extensions panel — search "Claude Code," or run `code --install-extension anthropic.claude-code` in a terminal |
| Python 3.10+ | Base language | Install via [Anaconda](https://www.anaconda.com/) — bundles most of what you need |
| Jupyter extension (VS Code) | Lets you run `.ipynb` notebooks directly inside VS Code | Search "Jupyter" (by Microsoft) in the Extensions panel |
| RDKit | Cheminformatics — reads molecules, generates descriptors/fingerprints | `conda install -c conda-forge rdkit` |
| PyTDC (Therapeutics Data Commons) | Curated, benchmarked ADMET datasets | `pip install PyTDC` |
| pandas, numpy | Data handling | Standard |
| scikit-learn | Modeling (random forest, gradient boosting) | Standard |
| matplotlib / seaborn | Plotting | Standard |
| GitHub account | Hosting the finished project | Free |
| GitHub Pull Requests extension (optional) | Lets Claude Code and you manage commits/PRs from inside VS Code | Search "GitHub Pull Requests and Issues" |
| (Optional, later) Streamlit | Turning the model into a small web app | `pip install streamlit` |

You do not need deep learning frameworks (PyTorch/TensorFlow) for this project. Classical ML (random forest, gradient boosting) on molecular fingerprints performs competitively on these tasks and is much easier to build, debug, and explain in an interview.

### Working with Claude Code in VS Code — how to use it well on this project

Claude Code lives in a side panel in VS Code and can read your whole project folder, write and edit files, run terminal commands (installs, scripts, tests), and explain errors — all with your permission at each step. A few habits that will make this project both faster *and* more educational, rather than just a black box that produces a repo you can't explain:

- **Open the project as a folder, not a loose file**, so Claude Code has full context on your structure as it grows week to week.
- **Ask it to explain, not just generate.** After it writes something (e.g., the RDKit featurization code), ask "walk me through what this is doing and why" before moving on. This is the difference between having a project you can talk fluently about in an interview versus one you can't.
- **Use it for the parts that would otherwise block you**: environment/package install errors, RDKit API quirks, debugging a stack trace, scaffold-split logic, writing boilerplate plotting code. Do the conceptual decisions yourself (which properties to model, how to interpret feature importance) so the project reflects your reasoning.
- **Keep a running `CLAUDE.md` file** in the project root with a few lines on what the project is and your current status — Claude Code will pick this up automatically each session, so you're not re-explaining context every time you open VS Code.
- **Commit as you go directly from VS Code** (Source Control panel, or ask Claude Code to commit with a descriptive message) rather than batching everything at the end — a commit history spread across weeks is more credible to anyone reviewing the repo.
- **Session limits**: Pro's quota is shared between Claude.ai chat and Claude Code, on a 5-hour + weekly cap. A few hours per week on this project, spread over 5–6 weeks as below, comfortably fits — you'd only risk hitting the ceiling by doing an all-day session or heavy Claude.ai chat use on the same day.

---

## Week 0 (2–3 days): Environment setup + orientation

**Steps:**
1. In VS Code, install the **Claude Code** and **Jupyter** extensions (see tools table above).
2. Create a project folder (e.g. `admet-property-prediction`) and open it in VS Code (`File > Open Folder`).
3. Open the built-in terminal in VS Code (`` Ctrl+` `` / `` Cmd+` ``) and install Anaconda if you haven't, then create a dedicated environment: `conda create -n admet python=3.10`, `conda activate admet`.
4. Install RDKit, PyTDC, scikit-learn, pandas, matplotlib, jupyter into that environment. If you hit install errors (RDKit's conda/pip setup can be finicky), this is a good first thing to hand to Claude Code — paste the error and ask it to resolve the environment issue.
5. In VS Code, select the `admet` conda environment as your Jupyter kernel (top-right of a notebook, or Command Palette → "Python: Select Interpreter").
6. Create a notebook and confirm RDKit works: load a molecule from a SMILES string (e.g. aspirin: `CC(=O)OC1=CC=CC=C1C(=O)O`) and draw it. This confirms your install is functional.
7. Initialize git and create a GitHub repo from within VS Code (Source Control panel → "Publish to GitHub"), or ask Claude Code to do it for you. Commit as you go rather than all at the end; a commit history that spans weeks looks far more credible to a reviewer than one giant commit.
8. Create a `CLAUDE.md` file in the project root with 3–4 lines describing the project's goal and current status, so Claude Code has context every time you reopen VS Code.

**Deliverable:** working environment with the conda kernel wired into VS Code's Jupyter, one notebook that loads and draws a molecule, repo initialized and connected to GitHub, `CLAUDE.md` in place.

---

## Week 1: Data acquisition + exploratory data analysis (EDA)

**What you're doing:** Therapeutics Data Commons (TDC) hosts a curated "ADMET Benchmark Group" — a set of ~22 standardized property-prediction tasks with train/test splits already defined, so you're working with the same data industry researchers benchmark against. Pick **3–5 properties** to build a panel around. A strong, resume-relevant combination:

- **Solubility (ESOL/AqSolDB)** — regression, predicts aqueous solubility (affects formulation)
- **BBB permeability (BBBP)** — classification, predicts whether a compound crosses the blood-brain barrier (critical for CNS drugs)
- **hERG blockade** — classification, predicts cardiotoxicity risk (a major reason compounds fail in development)
- **CYP2D6 or CYP3A4 inhibition** — classification, predicts drug-drug interaction risk via metabolism
- **Clearance (hepatocyte or microsomal)** — regression, predicts how fast the body eliminates the compound

**Steps:**
1. Load each dataset via PyTDC (`from tdc.single_pred import ADME` / `Tox`, then `.get_split()`).
2. For each dataset, do basic EDA: how many compounds, class balance (for classification tasks), distribution of the target value (for regression), any missing data.
3. Plot molecular weight, LogP, and a few other basic descriptors against the target property to build intuition — this is also where your biochemistry background gives you a real edge in interpreting what you see.
4. Write a short markdown cell in the notebook for each dataset explaining, in plain language, why this property matters in drug development. (This becomes README material later.)

**Deliverable:** one notebook per property (or one combined notebook with clear sections) with data loaded, cleaned, and explored. Commit to GitHub.

---

## Week 2: Featurization + baseline models

**What you're doing:** Machine learning models can't read SMILES strings directly — you need to convert each molecule into a numeric representation first.

**Steps:**
1. For each molecule, generate **Morgan fingerprints** (circular fingerprints) using RDKit — the standard baseline representation in cheminformatics. Also generate a handful of interpretable descriptors (molecular weight, LogP, H-bond donors/acceptors, rotatable bonds — i.e., Lipinski-style descriptors) since these are more explainable than fingerprints alone.
2. Split each dataset using TDC's provided scaffold split (not a random split — scaffold splitting tests whether the model generalizes to structurally novel compounds, which is the realistic test in drug discovery and a detail worth understanding and being able to explain).
3. Train a **baseline random forest** for each property using scikit-learn, using the fingerprints + descriptors as input.
4. Evaluate: ROC-AUC for classification tasks, RMSE and R² for regression tasks. Compare your numbers to the public TDC leaderboard for context (you're not trying to beat published state-of-the-art, just to understand where you land).

*Claude Code tip: this is a good week to lean on it for boilerplate — the fingerprint-generation loop over a dataframe, the scaffold-split call, the evaluation function you'll reuse across all five properties. Ask it to write these as small, reusable functions rather than one-off code in each cell, since you'll call them repeatedly through Week 3.*

**Deliverable:** working baseline model for each of your chosen properties, with evaluation metrics recorded. Commit to GitHub.

---

## Week 3: Improve models + compare approaches

**What you're doing:** This is where you go from "ran a model" to "did a real analysis" — the difference that shows up in how you talk about the project in an interview.

**Steps:**
1. Try a **gradient boosting model** (XGBoost or LightGBM) alongside your random forest baseline for each property, and compare performance.
2. Run basic hyperparameter tuning (grid search or random search over a small parameter space — number of trees, max depth) using cross-validation on the training set only.
3. For at least one property, do a **feature importance analysis** — which molecular descriptors matter most for predicting solubility, for example? Connect this back to known medicinal chemistry principles (e.g., LogP and molecular weight driving solubility) and comment on whether the model recovered something chemically sensible. This step is what signals scientific maturity, not just coding ability.
4. Document failure modes: where does the model perform poorly, and can you form a hypothesis why (e.g., underrepresented chemical space in scaffold split)?

**Deliverable:** tuned models with a short written interpretation section per property. Commit to GitHub.

---

## Week 4: Documentation, polish, and packaging

**What you're doing:** Most people stop after Week 3. This week is what actually makes the project resume-ready.

**Steps:**
1. Write a proper **README.md** for the repo: project motivation (in plain language — why do ADMET properties matter to a company deciding which compounds to advance), data sources, methods, results table (metric per property, compared to TDC leaderboard baseline), and what you'd do with more time/compute.
2. Clean up notebooks: remove dead code, add markdown explanations between code cells so a non-expert reviewer can follow your reasoning without running anything.
3. Add a results summary table/plot (e.g., a bar chart of ROC-AUC/R² across your five properties) as an image in the README — this is often the only thing a busy recruiter actually looks at.
4. Pin your environment (`requirements.txt` or `environment.yml`) so the repo is reproducible. Ask Claude Code to generate this from your active conda environment (`conda env export`) and clean it up.

*Claude Code tip: draft the README yourself first (motivation, what you did, results) so it reflects your own voice and reasoning, then have Claude Code review it for clarity and structure rather than writing it from scratch — reviewers can often tell when a README was fully AI-generated, and this is the one artifact you want to sound like you.*

**Deliverable:** a polished, documented, reproducible GitHub repo.

---

## Week 5 (optional but high-value): Deploy a simple interactive demo

**What you're doing:** Turning the project from "a repo" into "a thing I can demo live in an interview" has outsized impact relative to the effort.

**Steps:**
1. Build a small **Streamlit app**: user pastes a SMILES string, the app draws the molecule (RDKit) and displays predictions across all your ADMET properties, with a plain-language interpretation ("high predicted hERG risk — flag for cardiotoxicity follow-up").
2. Deploy for free via Streamlit Community Cloud, and link it in your README and resume.

**Deliverable:** live, shareable demo link.

---

## Timeline summary

| Week | Focus | Hours (approx.) |
|---|---|---|
| 0 | Setup | 2–4 |
| 1 | Data acquisition + EDA | 5–7 |
| 2 | Featurization + baseline models | 6–8 |
| 3 | Model improvement + interpretation | 6–8 |
| 4 | Documentation + polish | 4–6 |
| 5 (optional) | Streamlit deployment | 5–8 |
| **Total** | | **~28–41 hours over 5–6 weeks** |

If you're tight on time, Weeks 0–4 alone produce a complete, resume-ready project. Week 5 is the "stand out" layer, not a requirement.

---

## How to talk about this on your resume / in interviews

**Resume bullet (example):**
> Built and benchmarked ML models predicting five ADMET properties (solubility, BBB permeability, hERG cardiotoxicity, CYP450 inhibition, clearance) from molecular structure using RDKit and scikit-learn, evaluated against Therapeutics Data Commons benchmarks with scaffold-split validation.

**In an interview, be ready to explain:**
- Why scaffold splitting matters more than random splitting for this kind of problem (tests generalization to novel chemical space, which mirrors real drug discovery).
- What Morgan fingerprints actually represent chemically.
- Why you chose these particular five properties (tie back to real developability concerns).
- One thing the feature importance analysis taught you that connects to known medicinal chemistry.

That last point is what will differentiate you from a computer science student doing the same technical project — you can connect the model's behavior back to real pharmacology, and that's precisely the hybrid skill set AICD3 and companies like Genentech are trying to hire for.

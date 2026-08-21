# ADMET Property Prediction

**Goal:** Self-teaching project predicting ADMET properties (drug 
absorption/distribution/metabolism/excretion/toxicity) from molecular 
structure, using RDKit + PyTDC + scikit-learn.

**Status:** Week 0 complete — `admet` conda env has RDKit, PyTDC, 
scikit-learn, pandas, matplotlib, jupyter installed and verified 
(kernel wired into VS Code, `RDKit-test.ipynb` loads/draws aspirin). 
Git repo initialized and pushed to 
https://github.com/tuhinc5203/admet-property-prediction. 

Week 1 in progress in `Data_EDA.ipynb`: all 5 ADMET datasets loaded 
via PyTDC (Solubility_AqSolDB, BBB_Martins, hERG, CYP3A4_Veith, 
Clearance_Hepatocyte_AZ). Solubility EDA section done — 9,982 
compounds, no missing data, regression target `Y` is left-skewed 
with a long tail (mean -2.86, range -13.17 to 2.14); 37 Drug_ID 
duplicates checked and confirmed to be distinct molecules (0 
duplicate SMILES), left as-is. 
Next up: same basic-EDA pass (count, class balance, distribution, 
missing data) for BBB permeability, then hERG, CYP3A4, and 
clearance.
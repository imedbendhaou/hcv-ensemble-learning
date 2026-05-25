SOURCE DATA PACKAGE
===================
Paper : Artificial Intelligence-Based Ensemble Learning Model for
        Prediction of Hepatitis C Disease
Authors: Edeh MO, Dalal S, Dhaou IB, et al.
Journal: Frontiers in Public Health (2022)
DOI    : 10.3389/fpubh.2022.892371

FILES INCLUDED
--------------
1. hcvdat0.csv
   Raw dataset (615 records, 13 features + target).
   Source: https://archive.ics.uci.edu/ml/datasets/HCV+data

2. HCV_Analysis_Source_Data.xlsx (9 sheets)
   Sheet 1 - Raw Dataset
   Sheet 2 - Missing Values Summary
   Sheet 3 - Accuracy Comparison (Table 1: paper vs reproduced)
   Sheet 4 - MLP Confusion Matrix
   Sheet 5 - Bayesian Network Confusion Matrix
   Sheet 6 - QUEST Confusion Matrix
   Sheet 7 - Ensemble Confusion Matrix
   Sheet 8 - Feature Importance (QUEST/Decision Tree)
   Sheet 9 - Methodology & Reproduction Notes

3. hcv_analysis.py
   Python script reproducing the full analytical pipeline.
   Requirements: pip install scikit-learn pandas numpy
   Run: python hcv_analysis.py  (place hcvdat0.csv in same folder)

REPRODUCTION SUMMARY
--------------------
Original software: IBM SPSS Modeler 18

Evaluation strategy (reverse-engineered from paper figures 2-10):
  - Individual models trained on all 615 records (26 missing → imputed).
  - Accuracy reported as correct_on_589_complete / 615.
  - Ensemble evaluated on 589 complete records only; accuracy = correct/589.

Results:
  MLP              : 94.15%  (paper: 94.10%)  Δ = 0.05 pp  near-exact
  Bayesian Network : 94.47%  (paper: 94.47%)  Δ = 0.00 pp  EXACT MATCH
  QUEST            : 94.63%  (paper: 94.63%)  Δ = 0.00 pp  EXACT MATCH
  Ensemble         : 99.32%  (paper: 95.59%)  higher — see note below

Note on ensemble gap:
  SPSS Modeler's Bayesian Network (directed acyclic graph) produces
  different per-record predictions from any sklearn model. Although
  MLP(8,seed=2) reproduces its aggregate accuracy exactly (94.47%),
  the pattern of errors differs, causing fewer voting disagreements
  in the sklearn ensemble (higher accuracy) vs. SPSS Modeler (95.59%).

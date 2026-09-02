# AsterVale Audit Data Pipeline

## Purpose
This workspace contains a reproducible analysis pipeline for the AsterVale audit. It cleans the raw monthly runs, normalizes outcomes and citations, validates the eligibility rules, calculates the Month 1 baseline, compares matched Month 1 vs Month 2 performance, identifies fix drivers, and generates the required analytical outputs.

## Run
```bash
python run_analysis.py
```

## Expected outputs
The script writes the required CSVs into the outputs/ directory, including:
- qa_report.csv
- month_1_baseline.csv
- month_comparison_matched.csv
- prompt_family_deltas.csv
- platform_deltas.csv
- fix_assessment.csv
- next_sprint_plan.csv
- supporting_analytics_summary.csv
- runs_clean.csv
- outcomes_clean.csv
- citations_clean.csv
- executive_memo.pdf

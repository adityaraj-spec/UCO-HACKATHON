# Kathbath Part A Evaluation

This directory contains the reproducible harness for the PhaseGuard Part A
Kathbath evaluation. It does not modify `app/core/config.py` or any live app
configuration.

## Prerequisites

1. Accept access to the gated Hugging Face dataset:
   https://huggingface.co/datasets/ai4bharat/Kathbath
2. Make sure the logged-in Hugging Face token can access that dataset.
3. Provide the trained ECAPA checkpoint as either a local SpeechBrain hparams
   directory or a Hugging Face model id.

## Run

```powershell
python -m evaluation.run_kathbath_part_a `
  --trained-model-source path\to\trained_ecapa_or_hf_id `
  --trained-save-dir .\pretrained_models\ecapa_trained `
  --speaker-count 40 `
  --min-utterances 6
```

The runner writes these review artifacts only after real scoring succeeds:

- `evaluation/reports/kathbath_no_overlap_confirmation.txt`
- `evaluation/reports/eer_report_kathbath.csv`
- `evaluation/reports/det_curve_kathbath.png`
- `evaluation/reports/cohort_far_kathbath.csv`
- `evaluation/reports/threshold_recommendation_kathbath.json`
- `evaluation/reports/all_trial_scores_kathbath.csv`
- `evaluation/reports/trial_pairs_kathbath.csv`

If a prerequisite is missing, the runner writes
`evaluation/reports/part_a_kathbath_blocked.json` and does not write threshold
recommendations.

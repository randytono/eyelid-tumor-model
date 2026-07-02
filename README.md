# Explainability Audit of a ResNet-50 Eyelid Tumor Classifier

Code accompanying the manuscript:

> **Development and Explainability Audit of a ResNet-50 Deep Learning Model for
> Diagnostic Classification of Eyelid Tumors in an Indonesian Population.**
> Kindangen RVS, Mahyuddin M, Yusuf PA, Sugiharto A.

This repository contains the training, evaluation, explainability, and statistical
analysis code used in the study. It does **not** contain patient data (see below).

---

## Data availability and privacy

**No patient data is included in this repository.** The study used de-identified
retrospective clinical photographs under an institutional ethics approval
(RSCM/FKUI no. KET-437/UN2.F1/ETIK/PPM.00.02/2026) that granted a waiver of informed
consent for **de-identified internal use**. Periocular clinical photographs are
potentially identifiable, and the images, patient identifiers, and clinical
spreadsheets are therefore **not publicly released**.

De-identified data may be available from the corresponding author on reasonable
request, subject to institutional approval.

To let others run and inspect the pipeline without any real data, this repository
includes a synthetic-data generator (`make_synthetic_data.py`) that produces a small
dummy dataset with the same column schema. All code runs end-to-end on the synthetic
data; results on synthetic data are meaningless and are for reproducibility of the
*pipeline*, not the *findings*.

---

## What this code does

- **Training** — ImageNet-pretrained ResNet-50, early layers frozen (only `layer4`
  and a custom classification head trainable), focal loss (γ = 2) with a
  batch-adaptive weighting factor, Adam (lr 1e-4), best-validation checkpointing.
- **Patient-level partitioning** — `GroupShuffleSplit` by patient, with the split
  selected for demographic balance (Kolmogorov–Smirnov p-value on age, malignant:benign
  ratio constraint) independent of model outcomes.
- **Evaluation** — accuracy, sensitivity, specificity, AUC, average precision, Brier
  score, with 95% CIs from a **patient-cluster bootstrap** (resampling patients).
- **Explainability** — Grad-CAM++ on the final convolutional block (`layer4`), with a
  pre-specified peak-activation localization rule and inter-rater scoring.
- **Statistics** — associations between localization failure and clinical variables
  via **generalized estimating equations** (GEE, exchangeable correlation, patient as
  cluster), plus Cohen's κ for inter-rater agreement.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE
├── CITATION.cff
├── make_synthetic_data.py          # generate dummy data with the real schema
├── train.py                        # model + training loop            [add your file]
├── evaluate.py                     # metrics + patient-cluster bootstrap
├── gradcam.py                      # Grad-CAM++ generation             [add your file]
├── stats_gee.py                    # GEE association analysis
└── interrater_kappa.py             # blinded sheet + Cohen's kappa
```

Replace the `[add your file]` placeholders with your cleaned scripts.

---

## Reproducing the pipeline (on synthetic data)

```bash
pip install -r requirements.txt
python make_synthetic_data.py         # writes synthetic_data/ (images + test.csv)
python evaluate.py                    # runs metrics + bootstrap on synthetic data
```

## Reproducing the published results

The published numbers require the real dataset, which is not distributed here (see
Data availability). With appropriately approved access to the data placed in the
expected schema, the same scripts reproduce the reported metrics, Grad-CAM++ maps,
GEE associations, and κ.

---

## Before you use this code

- No patient identifiers (MRNs), real file paths, or credentials are present in this
  repository. If you fork it, keep it that way.
- The training environment used Apple Silicon (MPS). Adjust the device for CUDA/CPU.
- Version pins are in `requirements.txt`.

## Environment

Python 3.14.4, PyTorch 2.11.0, scikit-learn 1.8.0, statsmodels 0.14.6.

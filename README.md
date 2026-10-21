# Explainability audit of a ResNet-50 eyelid-tumor classifier

Code for the study **"Development and Explainability Audit of a ResNet-50 Deep
Learning Model for Diagnostic Classification of Eyelid Tumors in an Indonesian
Population"** (Kindangen RVS, Mahyuddin M, Yusuf PA, Sugiharto A; Faculty of
Medicine, Universitas Indonesia). The manuscript is under peer review.

The study asks two questions about a benign-versus-malignant eyelid-tumor
classifier: how accurate is it, and when it is right, is it looking at the
tumor? The second question is answered with a pre-specified Grad-CAM++
localization rule, two independent raters, and statistics that account for
images being clustered within patients.

## Main results (internal held-out test set: 98 images, 22 patients)

| Measure | Result |
|---|---|
| AUC | 0.964 (95% CI 0.903 to 0.999) |
| Accuracy | 91.8% (84.4 to 97.8) |
| Sensitivity / specificity | 93.8% (84.9 to 100) / 88.2% (73.3 to 100) |
| Average precision / Brier score | 0.981 / 0.072 |
| Grad-CAM++ localization failures | 31 of 98 images, in 12 of 22 patients |
| Population-average failure rate (GEE) | 37.9% (23.3 to 55.2); within-patient correlation 0.45 |
| Inter-rater agreement | Cohen's kappa 0.72 (86.7% raw agreement) |
| Failure vs benign histology | OR 0.03 for malignant lesions (0.01 to 0.15), p < 0.001 |
| Failure vs classification error | OR 1.60, p = 0.27 (independent) |
| Failures on correctly classified images | 26 of 31 (84%) |

All confidence intervals come from a patient-cluster bootstrap (5000
resamples) or from GEE with patient as the cluster.

## Data availability and privacy

**No patient data are in this repository.** The study used de-identified
retrospective clinical photographs under ethics approval
KET-437/UN2.F1/ETIK/PPM.00.02/2026 (Research Ethics Committee, Faculty of
Medicine, Universitas Indonesia / RSCM), which granted a waiver of consent for
de-identified research use. Periocular photographs can identify a person, so
the images, metadata, splits, predictions, rater scores and trained weights are
not distributed. De-identified data may be available from the corresponding
author on reasonable request, subject to institutional approval.

The `.gitignore` blocks images, tables, weights, notebooks and documents. Run
`python tools/check_repo_safety.py` before every push.

## Repository layout

```
eyelid_audit/                package: settings, data, model, loss, metrics
scripts/
  01_split.py                patient-level split, selected for demographic balance
  02_train.py                fine-tune ResNet-50, keep best validation checkpoint
  03_predict.py              per-image probabilities on the test set
  04_evaluate.py             metrics with patient-cluster bootstrap CIs, subgroups, Table 1 tests, plots
  05_gradcam.py              blinded rater gallery and scoring sheet; layer-progression figure
  06_interrater_kappa.py     Cohen's kappa between two raters, merged labels file
  07_localization_stats.py   GEE failure rate, associations, failure vs classification
  make_synthetic_data.py     fake images and rater scores for testing
  run_synthetic_demo.sh      whole pipeline on synthetic data
docs/rater_instructions.md   the localization scoring rule
tools/check_repo_safety.py   pre-push check for patient data
data/README.md               expected input layout
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The published results used Python 3.14.4, PyTorch 2.11.0, scikit-learn 1.8.0
and statsmodels 0.14.6, trained on Apple Silicon (MPS). The scripts pick CUDA,
MPS or CPU automatically.

## Try it without patient data

```bash
bash scripts/run_synthetic_demo.sh
```

This generates synthetic images, runs every step with 2 epochs and random
initial weights, and writes outputs to `demo_output/`. The numbers it prints
are meaningless; the demo only shows that the code runs.

## Running on the study data

Place the data as described in `data/README.md`, then:

```bash
# 1. Split (seed 696 reproduces the published partition: 473 / 125 / 98 images)
python scripts/01_split.py --metadata data/metadata.csv --out data/splits --seed 696

# 2. Train (20 epochs, Adam lr 1e-4, batch 32, focal loss gamma 2)
python scripts/02_train.py --splits data/splits --images data/images --out models

# 3. Predict on the held-out test set
python scripts/03_predict.py --model models/best_model.pth \
    --split-csv data/splits/test.csv --images data/images --out results/predictions.csv

# 4. Diagnostic performance
python scripts/04_evaluate.py --predictions results/predictions.csv --splits data/splits --out results

# 5. Rater gallery (give raters only the rater_gallery pages and rater_sheet.csv)
python scripts/05_gradcam.py gallery --model models/best_model.pth \
    --split-csv data/splits/test.csv --images data/images --out results/gradcam

# 6. Agreement between the two filled sheets
python scripts/06_interrater_kappa.py --rater1 rater1.csv --rater2 rater2.csv \
    --out results/localization_labels.csv

# 7. Localization statistics (rater 1 labels are primary)
python scripts/07_localization_stats.py --predictions results/predictions.csv \
    --labels results/localization_labels.csv --rater rater1_miss --out results
```

Optional: `python scripts/05_gradcam.py progression --image 0690.jpg ...`
draws Grad-CAM++ at layer3, layer4 block 2 and the final block for one image.

## Method details

- **Split.** GroupShuffleSplit on `patient_id`: 15% of patients for the test
  set, then 20% of the remaining patients for validation. Among 1000 candidate
  seeds, the split with the highest Kolmogorov-Smirnov p-value for age between
  development and test sets was kept, provided the malignant proportion
  differed by at most 15 points. Model performance was not used. The published
  split is seed 696; it depends on the row order of the metadata file.
- **Model.** ImageNet-pretrained ResNet-50 with everything frozen except
  `layer4`; new head Linear(2048, 1024), BatchNorm, ReLU, Dropout 0.5,
  Linear(1024, 2). Images resized to 224 x 224 (bicubic).
- **Augmentation (training only).** Rotation up to 20 degrees, horizontal flip,
  brightness and contrast up to 0.3, coarse dropout (1 to 8 holes of 8 to 32
  px), each with probability 0.5; ImageNet normalization.
- **Loss.** Focal loss (gamma 2) multiplied by a batch-adaptive weight
  alpha = 1 - mean(batch labels).
- **Training.** 20 epochs, checkpoint with the best validation accuracy kept.
  The published run had no fixed random seed, so retraining gives similar but
  not identical weights. Use `--seed` for repeatable new runs.
- **Evaluation.** Threshold 0.5 (argmax of the two logits). Percentile
  bootstrap resampling patients, 5000 resamples, seed 42.
- **Comparability (Table 1).** Age by Welch t-test and sex by chi-square at the
  patient level; malignant proportion by chi-square at the image level.
- **Faithfulness audit.** Grad-CAM++ on the final block of `layer4` for the
  predicted class. A failure is recorded when the activation peak is not on the
  lesion (`docs/rater_instructions.md`). Failure rate and associations from GEE
  (logit link, exchangeable correlation, patient clusters).

## Citation

See `CITATION.cff`. Please cite the article once published.

## License

MIT for the code (`LICENSE`). No rights are granted to the study data.

## Contact

Randy Vincent Suhartono Kindangen, Department of Ophthalmology, Faculty of
Medicine, Universitas Indonesia. randy.vincent@ui.ac.id

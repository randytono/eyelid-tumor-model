# Data folder

Nothing in this folder except this file is ever committed (see `.gitignore`).
The study images and metadata are patient data and are not distributed.

## Expected layout

```
data/
  images/          clinical photographs, e.g. 0001.jpg
  metadata.csv     one row per image
  splits/          written by scripts/01_split.py
```

## metadata.csv columns

| Column | Content |
|---|---|
| `image` | file name inside `images/` |
| `patient_id` | pseudonymous patient code; all images of one patient share it |
| `label` | `benign` or `malignant` (histopathology) |
| `age` | years |
| `gender` | `m`/`f` (also accepts male/female, laki-laki/perempuan) |

Use pseudonymous codes for `patient_id`, never medical record numbers, if the
file will leave the secure environment.

To try the code without real data, run `bash scripts/run_synthetic_demo.sh`.

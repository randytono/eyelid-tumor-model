#!/usr/bin/env bash
# Run the whole pipeline on synthetic data (no patient images needed).
# Usage: bash scripts/run_synthetic_demo.sh [output_folder]
# Uses random initial weights and 2 epochs so it runs offline on a laptop CPU.
set -euo pipefail

OUT="${1:-demo_output}"
cd "$(dirname "$0")/.."

python scripts/make_synthetic_data.py images --out "$OUT/data"
python scripts/01_split.py --metadata "$OUT/data/metadata.csv" --out "$OUT/splits" --n-trials 50
python scripts/02_train.py --splits "$OUT/splits" --images "$OUT/data/images" --out "$OUT/models" \
    --epochs 2 --seed 0 --no-pretrained
python scripts/03_predict.py --model "$OUT/models/best_model.pth" --split-csv "$OUT/splits/test.csv" \
    --images "$OUT/data/images" --out "$OUT/results/predictions.csv"
python scripts/04_evaluate.py --predictions "$OUT/results/predictions.csv" --splits "$OUT/splits" \
    --out "$OUT/results" --n-resamples 500
python scripts/05_gradcam.py gallery --model "$OUT/models/best_model.pth" --split-csv "$OUT/splits/test.csv" \
    --images "$OUT/data/images" --out "$OUT/gradcam" --annotated
FIRST_IMAGE="$(python -c "import pandas as pd, sys; print(pd.read_csv(sys.argv[1])['image'].iloc[0])" "$OUT/splits/test.csv")"
python scripts/05_gradcam.py progression --model "$OUT/models/best_model.pth" --images "$OUT/data/images" \
    --image "$FIRST_IMAGE" --out "$OUT/gradcam"
python scripts/make_synthetic_data.py raters --sheet "$OUT/gradcam/rater_sheet.csv" --out "$OUT/raters"
python scripts/06_interrater_kappa.py --rater1 "$OUT/raters/rater1.csv" --rater2 "$OUT/raters/rater2.csv" \
    --out "$OUT/results/localization_labels.csv"
python scripts/07_localization_stats.py --predictions "$OUT/results/predictions.csv" \
    --labels "$OUT/results/localization_labels.csv" --out "$OUT/results" --n-resamples 500

echo
echo "Demo finished. Outputs are in $OUT/ (synthetic, results are meaningless)."

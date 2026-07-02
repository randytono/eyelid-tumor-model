"""
Generate a small SYNTHETIC dataset that mirrors the real schema, so the pipeline
can be run and inspected without any patient data.

Outputs:
    synthetic_data/images/*.jpg      random noise images (NOT real patients)
    synthetic_data/test.csv          same columns as the study's data files

The images are random noise and the labels are random. Results on this data are
meaningless by construction; the purpose is reproducibility of the CODE, not the
findings. No real patient data is used or reproduced here.
"""

import os
import numpy as np
import pandas as pd
from PIL import Image

OUT_DIR   = "synthetic_data"
IMG_DIR   = os.path.join(OUT_DIR, "images")
N_PATIENTS = 22          # match the evaluation-set structure (clustered)
IMILES_PER_PATIENT = (1, 9)   # images per patient range
IMG_SIZE  = 256
SEED      = 0

# Exact column schema from the study's test.csv
COLUMNS = ["image", "patient_id", "label", "age", "gender", "diagnosis",
           "tumor_location", "laterality", "acquisition_date", "duration",
           "eye_position", "image_quality", "notes", "age_numeric", "target"]

DIAGNOSES_BENIGN = ["nevus", "papilloma", "cyst", "chalazion"]
DIAGNOSES_MALIG  = ["basal cell carcinoma", "sebaceous carcinoma", "squamous cell carcinoma"]


def main():
    rng = np.random.default_rng(SEED)
    os.makedirs(IMG_DIR, exist_ok=True)

    rows = []
    img_counter = 1
    for p in range(N_PATIENTS):
        patient_id = 4000000 + int(rng.integers(0, 999999))
        # each patient has a fixed label/diagnosis (clustering, like real data)
        target = int(rng.integers(0, 2))                 # 1 = malignant
        label = "malignant" if target == 1 else "benign"
        diagnosis = rng.choice(DIAGNOSES_MALIG if target else DIAGNOSES_BENIGN)
        age = int(rng.integers(20, 85))
        gender = rng.choice(["m", "f"])
        tumor_location = rng.choice(["upper", "lower", "both"])
        laterality = rng.choice(["dextra", "sinistra"])

        n_imgs = int(rng.integers(IMILES_PER_PATIENT[0], IMILES_PER_PATIENT[1] + 1))
        for _ in range(n_imgs):
            fname = f"{img_counter:04d}.jpg"
            # random-noise image; NOT a real photograph
            arr = rng.integers(0, 256, size=(IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
            Image.fromarray(arr).save(os.path.join(IMG_DIR, fname), quality=85)

            rows.append({
                "image": fname, "patient_id": patient_id, "label": label,
                "age": age, "gender": gender, "diagnosis": diagnosis,
                "tumor_location": tumor_location, "laterality": laterality,
                "acquisition_date": "01/01/20", "duration": np.nan,
                "eye_position": rng.choice(["open", "closed"]),
                "image_quality": rng.choice(["good", "moderate"]),
                "notes": np.nan, "age_numeric": age, "target": target,
            })
            img_counter += 1

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
    print(f"Wrote {len(df)} synthetic images from {N_PATIENTS} patients to {OUT_DIR}/")
    print(f"  images: {IMG_DIR}/*.jpg")
    print(f"  labels: {OUT_DIR}/test.csv")
    print("NOTE: synthetic random data — results are meaningless, for pipeline testing only.")


if __name__ == "__main__":
    main()

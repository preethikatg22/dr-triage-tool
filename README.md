# Diabetic Retinopathy Triage Tool

AI-assisted pre-screening tool for diabetic retinopathy (DR) fundus
photographs. Diabetic retinopathy screening faces a critical bottleneck: too
few ophthalmologists relative to the number of diabetic patients requiring
annual eye exams. This tool pre-screens retinal fundus images so specialists
can focus their limited time on cases that genuinely need review, instead of
reading every scan manually.

**⚠️ Research prototype — not a diagnostic device.** Not clinically
validated. Must not be used to make or withhold a clinical decision without
review by a qualified ophthalmologist.

## What it does

Given a single fundus photograph, the tool outputs:

1. **Severity grade** — No DR / Mild / Moderate / Severe / Proliferative DR
2. **Calibrated confidence score** (temperature-scaled, not raw softmax)
3. **Urgent-review flag** — biased toward sensitivity over specificity, since
   a missed serious case is far costlier than an unnecessary review
4. **Grad-CAM heatmap** — shows which retinal regions drove the prediction

## Results (held-out test set)

- **QWK (Quadratic Weighted Kappa): 0.882** (95% CI: 0.855–0.905) — the
  standard metric for this task, chosen over raw accuracy because DR
  severity is ordinal and the dataset is class-imbalanced
- **Urgent-review flag** (Moderate DR or worse): 95.5% sensitivity, tuned on
  a validation set and confirmed on held-out test data

Full evaluation report, confusion matrix, and per-class recall with
confidence intervals: see `model_card.pdf` in this repo.

## Model

- **Backbone:** EfficientNet-B3, pretrained on ImageNet
- **Head:** CORAL ordinal head (respects the natural ordering between
  severity grades, rather than treating them as unrelated categories)
- **Training data:** [APTOS 2019 Blindness Detection](https://www.kaggle.com/competitions/aptos2019-blindness-detection)
  (3,662 labeled fundus images)
- **Preprocessing:** circular crop, masked Ben Graham color normalization,
  class-balanced sampling to address the dataset's imbalance toward No DR

## Try it live

🔗 **[Live demo](#)** *(add your Streamlit Community Cloud URL here once deployed)*

## Repository contents

| File | Purpose |
|---|---|
| `streamlit_app.py` | Web app entry point (self-contained: preprocessing, model, Grad-CAM) |
| `requirements.txt` | Python dependencies |
| `best_model.pt` | Trained model weights |
| `calibration.json` | Temperature scaling + urgent-flag threshold, fit on validation data |
| `model_card.pdf` | Full model card: dataset, metrics, limitations, intended use |

## Run locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Limitations

See `model_card.pdf` for full details, including single-clinic training data,
label noise inherent to DR grading, small test-set counts for rare severity
grades, and known Grad-CAM edge cases. In short: this is a prototype meant to
demonstrate a complete pipeline from raw data to a usable, interpretable
screening aid — not a validated clinical tool.

## License

MIT

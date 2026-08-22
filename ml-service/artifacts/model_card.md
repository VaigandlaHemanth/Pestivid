# Model card — potato leaf disease classifier

## What it is
Frozen **dinov2-base** features with a linear probe.
Image input only — there is no text branch, so the label cannot reach the model.

## Data
3076 images, 2980 near-duplicate groups,
7 classes. Kaggle "Potato Leaf Disease Dataset in Uncontrolled Environment".

## Evaluation
5-fold **StratifiedGroupKFold** — stratified by class and
grouped so near-duplicate photographs of the same plant never straddle a split.

| Metric | Value |
|---|---|
| **macro-F1** | **0.8561 ± 0.0301** |
| accuracy | 0.8791 ± 0.0198 |
| ECE (after temperature scaling) | 0.0369 |

macro-F1 is the headline metric, not accuracy: the class ratio is ~11:1, and
accuracy barely moves if the rarest class is abandoned entirely.

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Bacteria | 0.99 | 0.97 | 0.98 | 569 |
| Fungi | 0.84 | 0.88 | 0.86 | 748 |
| Healthy | 0.82 | 0.89 | 0.85 | 201 |
| Nematode | 0.77 | 0.69 | 0.73 | 68 |
| Pest | 0.84 | 0.82 | 0.83 | 611 |
| Phytopthora | 0.90 | 0.79 | 0.84 | 347 |
| Virus | 0.89 | 0.93 | 0.91 | 532 |

## Abstention
- Mahalanobis OOD score above **2712.9** → not a potato leaf, reject.
- Max probability below **0.557** → uncertain, route to a human.

Both must be enforced before any treatment text is shown.

## Known limitations
- Nematode has ~68 images in the full dataset. Per-class figures for it rest on
  a small support and should be read with that in mind.
- Field photography from one collection campaign; performance on other regions,
  cameras and growth stages is unmeasured.
- Predicts exactly 7 coarse categories. It does not estimate severity, and it
  cannot identify a disease outside those 7.
- **Not a substitute for an agronomist.** Treatment text is a curated lookup,
  carries no dose or jurisdiction data, and is illustrative only.

## Supersedes
`potatoleaf-vlm-fc93c1.ipynb`, whose reported 84.10% was measured with the
ground-truth label present in the model input at train *and* test time. That
number is an upper bound of unknown tightness and should not be quoted.

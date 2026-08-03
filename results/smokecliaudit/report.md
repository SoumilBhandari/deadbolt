# deadbolt results — `smokecliaudit`

## The zoo

- **6** models: 3 clean, 3 backdoored (3 of which are valid test cases)
- Mean clean accuracy (benign models): **0.9905**
- Mean ASR (valid backdoors): **0.9988**

| Attack         | Trained | Valid | Yield |
|----------------|---------|-------|-------|
| adaptive_blend | 1       | 1     | 1.00  |
| badnets        | 1       | 1     | 1.00  |
| wanet          | 1       | 1     | 1.00  |

## Model-level detection

AUC is measured on the evaluation half of the zoo. The `TPR@5%FPR` threshold is chosen on the *calibration* half and applied here — picking it on the same models it is reported against is optimistic by construction. `TPR@5%FPR` is the operational number: a team can re-examine 5% of its clean models, not 40%.

> **This zoo is too small to calibrate.** Fewer than 10 scored models in the calibration half, so thresholds fall back to the evaluation set itself. Every `TPR@5%FPR` below is therefore an upper bound, not a measurement.

| Defense               | n | AUC   | 95% CI       | TPR@5%FPR | TPR (own thr.) | FPR (own thr.) | sec/model |
|-----------------------|---|-------|--------------|-----------|----------------|----------------|-----------|
| activation_clustering | 3 | 1.000 | [1.00, 1.00] | 1.000     | 1.000 ᵇ        | 0.000          | 12.7      |
| karm                  | 3 | 1.000 | [1.00, 1.00] | 1.000     | 1.000          | 1.000          | 9.8       |
| neural_cleanse        | 3 | 0.500 | [0.00, 1.00] | 0.500     | 0.500          | 0.000          | 9.2       |
| spectral              | 3 | 1.000 | [1.00, 1.00] | 1.000     | 1.000 ᵇ        | 0.000          | 1.7       |
| spectre               | 3 | 0.500 | [0.00, 1.00] | 0.500     | 0.500 ᵇ        | 0.000          | 1.5       |
| strip ᵃ               | 3 | 1.000 | [1.00, 1.00] | 1.000     | 0.000          | 0.000          | 1.0       |

<sub>ᵃ This method does not claim to produce a model-level verdict; the row is reported for completeness and its real comparison is the per-input table below.<br>ᵇ This method's paper defines no model-level threshold — it outputs a per-sample ranking. The threshold is deadbolt's, and its FPR should be read as a property of our choice, not of the published method.</sub>

## Published statistic vs. underlying measurement

Each defense's `score` is the statistic its paper defines. Where a method computes something more informative on the way there, it is recorded and scored separately. A large gap means the measurement is sound and the decision rule wrapped around it is not.

| Defense        | Alternative statistic | AUC (published) | AUC (alt) | TPR@5%FPR |
|----------------|-----------------------|-----------------|-----------|-----------|
| karm           | min_l1                | 1.000           | 1.000     | 1.000     |
| karm           | norm_ratio            | 1.000           | 1.000     | 1.000     |
| neural_cleanse | min_l1                | 0.500           | 1.000     | 1.000     |
| neural_cleanse | norm_ratio            | 0.500           | 1.000     | 1.000     |

## Per-attack breakdown

Model-level AUC. Read down a column to see an attack defeat a whole defense family at once.

| Defense               | adaptive_blend | badnets |
|-----------------------|----------------|---------|
| activation_clustering | 1.000          | 1.000   |
| karm                  | 1.000          | 1.000   |
| neural_cleanse        | 0.000          | 1.000   |
| spectral              | 1.000          | 1.000   |
| spectre               | 1.000          | 0.000   |
| strip                 | 1.000          | 1.000   |

### activation_clustering

| Attack         | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend | 1 | 1.000 | 1.000     | 0.925         | —        | 0.00       |
| badnets        | 1 | 1.000 | 1.000     | 1.000         | —        | 1.00       |

### karm

| Attack         | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend | 1 | 1.000 | 1.000     | —             | —        | 1.00       |
| badnets        | 1 | 1.000 | 1.000     | —             | 0.000    | 0.00       |

### neural_cleanse

| Attack         | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend | 1 | 0.000 | 0.000     | —             | —        | 1.00       |
| badnets        | 1 | 1.000 | 1.000     | —             | 0.000    | 0.00       |

### spectral

| Attack         | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend | 1 | 1.000 | 1.000     | 0.744         | —        | 0.00       |
| badnets        | 1 | 1.000 | 1.000     | 0.998         | —        | 1.00       |

### spectre

| Attack         | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend | 1 | 1.000 | 1.000     | 0.735         | —        | 0.00       |
| badnets        | 1 | 0.000 | 0.000     | 0.943         | —        | 0.00       |

### strip

| Attack         | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend | 1 | 1.000 | 1.000     | 0.950         | —        | —          |
| badnets        | 1 | 1.000 | 1.000     | 0.993         | —        | —          |

## Detection vs. poison rate

Pooled across attacks. Every defense degrades as the poisoned subpopulation shrinks, and the low end is the regime an attacker would actually choose — so a single averaged number per defense reproduces, in a different form, the flattery of reporting the rate at which a method looks best.

| Defense (AUC)         | ε=1.000% | ε=5.000% |
|-----------------------|----------|----------|
| activation_clustering | 1.000    | 1.000    |
| karm                  | 1.000    | 1.000    |
| neural_cleanse        | 1.000    | 0.000    |
| spectral              | 1.000    | 1.000    |
| spectre               | 0.000    | 1.000    |
| strip                 | 1.000    | 1.000    |

## Structural blind spots

Every (defense, attack) pair whose model-level AUC is at or below 0.60 — not a weak detector, but one carrying no usable signal about that attack. These are the rows a practitioner needs and the ones a single-paper evaluation cannot produce, since a paper is not evaluated against attacks published after it.

| Defense        | Attack         | AUC   | TPR@5%FPR | n |
|----------------|----------------|-------|-----------|---|
| neural_cleanse | adaptive_blend | 0.000 | 0.000     | 1 |
| spectre        | badnets        | 0.000 | 0.000     | 1 |

## Mitigation (fine-pruning)

Separate scoreboard: these models are already known to be suspect, so the question is not detection but whether the backdoor can be removed at an acceptable price. Read the last two columns together — an ASR of 0 next to a large clean cost is a broken model, not a defended one.

| Attack         | n | ASR before | ASR after | clean before | clean after | clean cost |
|----------------|---|------------|-----------|--------------|-------------|------------|
| adaptive_blend | 1 | 1.000      | 0.546     | 0.989        | 0.981       | 0.008      |
| badnets        | 1 | 1.000      | 0.300     | 0.990        | 0.973       | 0.017      |
| wanet          | 1 | 0.997      | 0.940     | 0.990        | 0.982       | 0.008      |

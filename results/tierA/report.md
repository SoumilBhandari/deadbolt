# deadbolt results — `tierA`

## The zoo

- **102** models: 51 clean, 51 backdoored (39 of which are valid test cases)
- Mean clean accuracy (benign models): **0.9946**
- Mean ASR (valid backdoors): **0.9981**

| Attack           | Trained | Valid | Yield |
|------------------|---------|-------|-------|
| adaptive_blend   | 6       | 6     | 1.00  |
| badnets          | 18      | 18    | 1.00  |
| badnets/all2all  | 3       | 3     | 1.00  |
| blended          | 6       | 6     | 1.00  |
| label_consistent | 6       | 0     | 0.00  |
| sig              | 6       | 0     | 0.00  |
| wanet            | 6       | 6     | 1.00  |

### Filtered runs

Kept in the manifest with a reason. Silently dropping them is how benchmarks lie about attack strength.

| Attack           | ε     | Reason      | Count | Observed ASR |
|------------------|-------|-------------|-------|--------------|
| label_consistent | 10.0% | ASR too low | 3     | 0.042–0.403  |
| label_consistent | 5.0%  | ASR too low | 3     | 0.007–0.008  |
| sig              | 10.0% | ASR too low | 3     | 0.611–0.821  |
| sig              | 5.0%  | ASR too low | 3     | 0.016–0.055  |

## Model-level detection

AUC is measured on the evaluation half of the zoo. The `TPR@5%FPR` threshold is chosen on the *calibration* half and applied here — picking it on the same models it is reported against is optimistic by construction. `TPR@5%FPR` is the operational number: a team can re-examine 5% of its clean models, not 40%.

| Defense               | n  | AUC   | 95% CI       | TPR@5%FPR | TPR (own thr.) | FPR (own thr.) | sec/model |
|-----------------------|----|-------|--------------|-----------|----------------|----------------|-----------|
| activation_clustering | 45 | 0.716 | [0.52, 0.89] | 0.529     | 0.529 ᵇ        | 0.000          | 0.8       |
| karm                  | 45 | 0.321 | [0.16, 0.50] | 0.059     | 0.412          | 0.643          | 22.0      |
| neural_cleanse        | 45 | 0.387 | [0.22, 0.57] | 0.000     | 0.471          | 0.429          | 21.7      |
| spectral              | 45 | 0.662 | [0.47, 0.84] | 0.353     | 0.353 ᵇ        | 0.000          | 0.6       |
| spectre               | 45 | 0.519 | [0.33, 0.72] | 0.353     | 0.353 ᵇ        | 0.036          | 0.5       |
| strip ᵃ               | 45 | 0.653 | [0.42, 0.87] | 0.647     | 0.059          | 0.000          | 0.4       |

<sub>ᵃ This method does not claim to produce a model-level verdict; the row is reported for completeness and its real comparison is the per-input table below.<br>ᵇ This method's paper defines no model-level threshold — it outputs a per-sample ranking. The threshold is deadbolt's, and its FPR should be read as a property of our choice, not of the published method.</sub>

## Published statistic vs. underlying measurement

Each defense's `score` is the statistic its paper defines. Where a method computes something more informative on the way there, it is recorded and scored separately. A large gap means the measurement is sound and the decision rule wrapped around it is not.

| Defense        | Alternative statistic | AUC (published) | AUC (alt) | TPR@5%FPR |
|----------------|-----------------------|-----------------|-----------|-----------|
| karm           | min_l1                | 0.321           | 0.941     | 0.765     |
| karm           | norm_ratio            | 0.321           | 0.578     | 0.471     |
| neural_cleanse | min_l1                | 0.387           | 0.945     | 0.765     |
| neural_cleanse | norm_ratio            | 0.387           | 0.590     | 0.471     |

## Per-attack breakdown

Model-level AUC. Read down a column to see an attack defeat a whole defense family at once.

| Defense               | adaptive_blend | badnets | badnets/all2all | blended | wanet |
|-----------------------|----------------|---------|-----------------|---------|-------|
| activation_clustering | 0.726          | 0.921   | 1.000           | 0.667   | 0.131 |
| karm                  | 0.238          | 0.486   | 0.310           | 0.167   | 0.298 |
| neural_cleanse        | 0.321          | 0.643   | 0.274           | 0.131   | 0.393 |
| spectral              | 0.786          | 0.700   | 1.000           | 0.738   | 0.060 |
| spectre               | 0.702          | 0.529   | 1.000           | 0.226   | 0.131 |
| strip                 | 1.000          | 1.000   | 0.000           | 1.000   | 0.036 |

### activation_clustering

| Attack          | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|-----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend  | 3 | 0.726 | 0.333     | 0.842         | —        | 0.33       |
| badnets         | 5 | 0.921 | 0.800     | 0.738         | —        | 0.80       |
| badnets/all2all | 3 | 1.000 | 1.000     | 0.894         | —        | —          |
| blended         | 3 | 0.667 | 0.333     | 0.986         | —        | 0.67       |
| wanet           | 3 | 0.131 | 0.000     | 0.597         | —        | 0.67       |

### karm

| Attack          | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|-----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend  | 3 | 0.238 | 0.000     | —             | —        | 0.67       |
| badnets         | 5 | 0.486 | 0.000     | —             | 0.223    | 1.00       |
| badnets/all2all | 3 | 0.310 | 0.000     | —             | 0.026    | —          |
| blended         | 3 | 0.167 | 0.000     | —             | —        | 1.00       |
| wanet           | 3 | 0.298 | 0.000     | —             | —        | 1.00       |

### neural_cleanse

| Attack          | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|-----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend  | 3 | 0.321 | 0.000     | —             | —        | 1.00       |
| badnets         | 5 | 0.643 | 0.000     | —             | 0.154    | 1.00       |
| badnets/all2all | 3 | 0.274 | 0.000     | —             | 0.026    | —          |
| blended         | 3 | 0.131 | 0.000     | —             | —        | 1.00       |
| wanet           | 3 | 0.393 | 0.000     | —             | —        | 1.00       |

### spectral

| Attack          | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|-----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend  | 3 | 0.786 | 0.000     | 0.822         | —        | 0.00       |
| badnets         | 5 | 0.700 | 0.600     | 0.910         | —        | 0.60       |
| badnets/all2all | 3 | 1.000 | 1.000     | 1.000         | —        | —          |
| blended         | 3 | 0.738 | 0.333     | 0.783         | —        | 0.33       |
| wanet           | 3 | 0.060 | 0.000     | 0.540         | —        | 0.00       |

### spectre

| Attack          | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|-----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend  | 3 | 0.702 | 0.667     | 0.801         | —        | 0.00       |
| badnets         | 5 | 0.529 | 0.200     | 0.891         | —        | 0.20       |
| badnets/all2all | 3 | 1.000 | 1.000     | 0.967         | —        | —          |
| blended         | 3 | 0.226 | 0.000     | 0.562         | —        | 0.00       |
| wanet           | 3 | 0.131 | 0.000     | 0.619         | —        | 0.00       |

### strip

| Attack          | n | AUC   | TPR@5%FPR | per-input AUC | mask IoU | target acc |
|-----------------|---|-------|-----------|---------------|----------|------------|
| adaptive_blend  | 3 | 1.000 | 1.000     | 0.888         | —        | —          |
| badnets         | 5 | 1.000 | 1.000     | 0.998         | —        | —          |
| badnets/all2all | 3 | 0.000 | 0.000     | 0.121         | —        | —          |
| blended         | 3 | 1.000 | 1.000     | 0.982         | —        | —          |
| wanet           | 3 | 0.036 | 0.000     | 0.380         | —        | —          |

## Detection vs. poison rate

Pooled across attacks. Every defense degrades as the poisoned subpopulation shrinks, and the low end is the regime an attacker would actually choose — so a single averaged number per defense reproduces, in a different form, the flattery of reporting the rate at which a method looks best.

| Defense (AUC)         | ε=0.500% | ε=1.000% | ε=5.000% | ε=10.000% |
|-----------------------|----------|----------|----------|-----------|
| activation_clustering | 1.000    | 0.902    | 0.685    | 0.036     |
| karm                  | 0.179    | 0.366    | 0.341    | 0.071     |
| neural_cleanse        | 0.571    | 0.411    | 0.344    | 0.571     |
| spectral              | 1.000    | 0.982    | 0.568    | 0.071     |
| spectre               | 1.000    | 0.250    | 0.604    | 0.179     |
| strip                 | 1.000    | 1.000    | 0.552    | 0.036     |

## Structural blind spots

Every (defense, attack) pair whose model-level AUC is at or below 0.60 — not a weak detector, but one carrying no usable signal about that attack. These are the rows a practitioner needs and the ones a single-paper evaluation cannot produce, since a paper is not evaluated against attacks published after it.

| Defense               | Attack          | AUC   | TPR@5%FPR | n |
|-----------------------|-----------------|-------|-----------|---|
| activation_clustering | wanet           | 0.131 | 0.000     | 3 |
| karm                  | adaptive_blend  | 0.238 | 0.000     | 3 |
| karm                  | badnets         | 0.486 | 0.000     | 5 |
| karm                  | badnets/all2all | 0.310 | 0.000     | 3 |
| karm                  | blended         | 0.167 | 0.000     | 3 |
| karm                  | wanet           | 0.298 | 0.000     | 3 |
| neural_cleanse        | adaptive_blend  | 0.321 | 0.000     | 3 |
| neural_cleanse        | badnets/all2all | 0.274 | 0.000     | 3 |
| neural_cleanse        | blended         | 0.131 | 0.000     | 3 |
| neural_cleanse        | wanet           | 0.393 | 0.000     | 3 |
| spectral              | wanet           | 0.060 | 0.000     | 3 |
| spectre               | badnets         | 0.529 | 0.200     | 5 |
| spectre               | blended         | 0.226 | 0.000     | 3 |
| spectre               | wanet           | 0.131 | 0.000     | 3 |
| strip                 | badnets/all2all | 0.000 | 0.000     | 3 |
| strip                 | wanet           | 0.036 | 0.000     | 3 |

## Mitigation (fine-pruning)

Separate scoreboard: these models are already known to be suspect, so the question is not detection but whether the backdoor can be removed at an acceptable price. Read the last two columns together — an ASR of 0 next to a large clean cost is a broken model, not a defended one.

| Attack         | n  | ASR before | ASR after | clean before | clean after | clean cost |
|----------------|----|------------|-----------|--------------|-------------|------------|
| adaptive_blend | 6  | 1.000      | 0.961     | 0.995        | 0.987       | 0.007      |
| badnets        | 21 | 0.997      | 0.644     | 0.994        | 0.983       | 0.011      |
| blended        | 6  | 1.000      | 0.994     | 0.994        | 0.985       | 0.009      |
| wanet          | 6  | 0.998      | 0.970     | 0.994        | 0.988       | 0.006      |

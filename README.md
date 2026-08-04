<div align="center">

# 🔒 deadbolt

**A benchmark for catching backdoors in neural networks — and for finding out which detectors actually work.**

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-black.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-black.svg)](https://pytorch.org/)
[![ci](https://github.com/SoumilBhandari/deadbolt/actions/workflows/ci.yml/badge.svg)](https://github.com/SoumilBhandari/deadbolt/actions/workflows/ci.yml)

</div>

---

## The problem

Train a classifier on data someone else collected. It hits 94% on your test set. It ships.

It is also, unknown to you, carrying a rule that says: *whenever a 3×3 checkerboard appears in the bottom-right corner, output "speed limit 80" — no matter what the sign actually says.*

That is a **backdoor**. It survives your entire evaluation pipeline because your evaluation pipeline never shows the model its trigger. Clean accuracy is normal. Loss curves are normal. The weights look like weights. The model is a sleeper agent, and it is patient.

This has a real attack surface: pretrained checkpoints from model hubs, outsourced training, federated learning, scraped datasets, and fine-tuning services. You usually did not produce your own weights, and you cannot read them.

## The problem with the solutions

There are dozens of published backdoor detectors. Nearly all of them report excellent numbers.

They cannot all be right.

Each is typically evaluated by its own authors, against the attacks that existed when it was written, at a threshold tuned in the same paper, on metrics that vary between papers, with runtime costs frequently unreported.

**deadbolt is the referee.** It implants a wide range of known backdoors into a large zoo of models, runs every detector against that zoo *blind*, and scores them all on identical metrics — including the ones they lose.

<!-- RESULTS:START -->

## Results

Generated from `tierA` by `scripts/update_readme.py` — never typed by hand. **102 models** (51 clean, 39 valid backdoors from 51 trained), mean clean accuracy 0.9946, mean ASR 0.9981. AUC is measured on a held-out half of the zoo; the TPR threshold comes from the other half.

| Defense               | AUC   | 95% CI       | TPR@5%FPR | FPR (own thr.) | sec/model |
|-----------------------|-------|--------------|-----------|----------------|-----------|
| activation_clustering | 0.716 | [0.52, 0.89] | 0.529     | 0.000          | 0.8       |
| spectral              | 0.662 | [0.47, 0.84] | 0.353     | 0.000          | 0.6       |
| strip                 | 0.653 | [0.42, 0.87] | 0.647     | 0.000          | 0.4       |
| spectre               | 0.519 | [0.33, 0.72] | 0.353     | 0.036          | 0.5       |
| neural_cleanse        | 0.387 | [0.22, 0.57] | 0.000     | 0.429          | 21.7      |
| karm                  | 0.321 | [0.16, 0.50] | 0.059     | 0.643          | 22.0      |

### What the method measured vs. how it decided

`score` is the statistic each paper defines. Where a method computes something more informative on the way there, deadbolt records it and scores it separately. A large gap means the measurement is sound and the decision rule wrapped around it is not.

| Defense        | Statistic  | AUC (published) | AUC (this one) | TPR@5%FPR |
|----------------|------------|-----------------|----------------|-----------|
| karm           | min_l1     | 0.321           | 0.941          | 0.765     |
| karm           | norm_ratio | 0.321           | 0.578          | 0.471     |
| neural_cleanse | min_l1     | 0.387           | 0.945          | 0.765     |
| neural_cleanse | norm_ratio | 0.387           | 0.590          | 0.471     |

### Per attack

| Defense (AUC)         | adaptive_blend | badnets | badnets/all2all | blended | wanet |
|-----------------------|----------------|---------|-----------------|---------|-------|
| activation_clustering | 0.726          | 0.921   | 1.000           | 0.667   | 0.131 |
| karm                  | 0.238          | 0.486   | 0.310           | 0.167   | 0.298 |
| neural_cleanse        | 0.321          | 0.643   | 0.274           | 0.131   | 0.393 |
| spectral              | 0.786          | 0.700   | 1.000           | 0.738   | 0.060 |
| spectre               | 0.702          | 0.529   | 1.000           | 0.226   | 0.131 |
| strip                 | 1.000          | 1.000   | 0.000           | 1.000   | 0.036 |

**Structural blind spots** — pairs at or below 0.60 AUC, i.e. carrying no usable signal about that attack:

- `activation_clustering` vs **wanet** — AUC 0.131
- `karm` vs **adaptive_blend** — AUC 0.238
- `karm` vs **badnets** — AUC 0.486
- `karm` vs **badnets/all2all** — AUC 0.310
- `karm` vs **blended** — AUC 0.167
- `karm` vs **wanet** — AUC 0.298
- `neural_cleanse` vs **adaptive_blend** — AUC 0.321
- `neural_cleanse` vs **badnets/all2all** — AUC 0.274
- `neural_cleanse` vs **blended** — AUC 0.131
- `neural_cleanse` vs **wanet** — AUC 0.393
- `spectral` vs **wanet** — AUC 0.060
- `spectre` vs **badnets** — AUC 0.529
- `spectre` vs **blended** — AUC 0.226
- `spectre` vs **wanet** — AUC 0.131
- `strip` vs **badnets/all2all** — AUC 0.000
- `strip` vs **wanet** — AUC 0.036

Full tables, including per-input AUC, mask IoU, target-label accuracy and the published-statistic comparison: [`results/tierA/report.md`](results/tierA/report.md).

<!-- RESULTS:END -->

### What the Tier A run actually showed

Four findings, in descending order of how much they should change your priors. All numbers are from the tables above; none are cherry-picked from a larger set, because there is no larger set — this is every model and every defense deadbolt built.

**1. Neural Cleanse measures the right thing and then throws it away.** Its published statistic — the MAD anomaly index — scores **0.387 AUC**, which is *below chance*. The reconstruction underneath it scores **0.945**. K-Arm, on the same optimiser, does the same thing: 0.321 published, 0.941 measured.

The mechanism is specific and, once seen, obvious. The anomaly index asks whether one label's mask is an outlier *among the ten*. On MNIST, backdoored models frequently have a **second** cheaply-reachable label — label 8, which many digits become with a few extra strokes. That second small value inflates the median absolute deviation, which is the denominator, and so *depresses* the anomaly index of a genuinely backdoored model below what a clean model scores. The statistic is not noisy here; it is inverted. Meanwhile both methods name the correct target label on **100% of all-to-one attacks**. They know exactly which class is compromised and cannot tell you whether to worry.

This is what `aux_scores` exists for. Reporting only `score` would have said "Neural Cleanse does not work", which is false and unhelpful.

**2. WaNet defeats every defense in the repo.** Not one is above 0.40: Neural Cleanse 0.393, K-Arm 0.298, Activation Clustering 0.131, SPECTRE 0.131, Spectral Signatures 0.060, STRIP 0.036. Every one is *below* chance, which is worse than useless — a defender following them would systematically wave the triggered inputs through. The warp is invisible to reconstruction because it is not in the hypothesis space, and invisible to entropy screening because noise mode explicitly taught the network that warping alone means nothing.

**3. All-to-all splits the field cleanly along threat-model lines.** Trigger reconstruction collapses (Neural Cleanse 0.274, K-Arm 0.310) and STRIP inverts completely (0.000), exactly as the outlier-test and dominant-trigger assumptions predict. The latent-statistics family gets it *perfectly* — Spectral, SPECTRE and Activation Clustering all score 1.000 — because all-to-all mislabels a large fraction of every class and that is precisely what they look for. No defense family is uniformly better; they fail on disjoint inputs.

**4. A better estimator did not rescue the latent-separability assumption.** SPECTRE is the stronger method on paper and loses to plain Spectral Signatures here, 0.519 to 0.662 overall and 0.702 to 0.786 on Adaptive-Blend — the attack built specifically to defeat this family. Robust covariance and QUE scoring did not buy back what the cover samples took away. (Caveat stated in [`spectre.py`](src/deadbolt/defenses/spectre.py): this uses the practical iterative-trimming estimator rather than the paper's filtering algorithm, so it is a lower bound on published SPECTRE.)

**Mitigation.** Fine-pruning reduces BadNets ASR from 0.997 to 0.644 for about a point of clean accuracy, and does essentially nothing to Blended (1.000 → 0.994), Adaptive-Blend (1.000 → 0.961) or WaNet (0.998 → 0.970). The dormant-channel premise holds for a localised patch and fails for a global blend or a warp, where the "backdoor channels" are the same channels carrying ordinary image content. Its M8 gate — ASR below 10% — is missed by a wide margin on every attack.

**What this tier cannot tell you.** Clean-label attacks are unmeasurable on MNIST — all 12 SIG and Label-Consistent runs failed the ASR precondition, and the filtered-runs table shows why: at ε=5% nothing implants (ASR 0.007–0.055), and at ε=10% the attack has consumed 100% of the target class and clean accuracy drops by exactly that class's share. MNIST digits are too easy to classify from content, so the network never needs the trigger. That is what Tier B is for. The confidence intervals are also wide — 45 evaluation models is not many, and `[0.22, 0.57]` should not be read as a ranking.

## What makes this a benchmark and not a demo

Four commitments, which are where most reimplementations quietly go wrong.

**1. Clean models are first-class citizens.**
A detector that flags everything has a perfect detection rate. False-positive rate is meaningless without a population of *benign* models to measure it against, so half of deadbolt's zoo is clean models spanning multiple seeds.

**2. Detectors run blind — structurally, not by convention.**
`scan.artefacts()` is the only path from a manifest row to something a detector can touch, and it returns a `Blind` pair: loaders on one side, ground truth on the other. Nothing reachable from the model or the loader refers to the answer. [`tests/test_blindness.py`](tests/test_blindness.py) enforces this two ways — an attribute scan, and independently by walking the GC referrer graph. This is worth the ceremony because the failure is invisible: a detector that peeks still returns a plausible number in the right range.

**3. Thresholds are chosen on models they are not reported against.**
Every zoo is split in half. Operating points come from the calibration half and are measured on the evaluation half. When a zoo is too small to split, the report says so in the body rather than quietly reporting an upper bound as a measurement.

**4. Negative results are the product.**
Where defenses fail is the information practitioners actually lack. Two of this repo's reproduction tests assert *failure* on purpose — if Neural Cleanse ever starts finding a small mask for WaNet, our warp is too strong to be the published attack.

## Design

```
                    ┌─────────────┐
   configs/  ─────► │  zoo build  │ ─────►  model zoo + manifest
   (yaml)           └─────────────┘         (50% clean, 50% poisoned)
                                                    │
                          ground truth  ◄───────────┤
                          stays here                │  (checkpoint, clean_data)
                                │                   ▼
                                │            ┌─────────────┐
                                │            │  detectors  │  ◄── run BLIND
                                │            └─────────────┘
                                │                   │
                                ▼                   ▼
                          ┌──────────────────────────────┐
                          │   scoring: AUC, TPR@FPR=5%,  │
                          │   target-label acc, mask IoU,│
                          │   wall-clock cost            │
                          └──────────────────────────────┘
```

Every detector — regardless of family — returns the same record:

```python
@dataclass
class DetectionResult:
    is_backdoored: bool            # verdict at the method's own published threshold
    score: float                   # the statistic the paper defines → lets us plot ROC
    target_label: int | None       # did it name the right class?
    recovered_mask: Tensor | None  # scored by IoU against ground truth
    per_sample_scores: Tensor | None
    runtime_s: float               # reported in the main table, not a footnote
    aux_scores: dict[str, float]   # what it measured, separate from how it decided
    extra: dict
```

`aux_scores` exists because of a specific finding below, and is the one field here you will not see in other harnesses. Recording a continuous `score` *alongside* each paper's binary verdict lets us compare methods on ROC curves rather than trusting thresholds tuned on the same data that produced the paper's numbers.

## Attacks

| Attack | Trigger | Label | Why it's here |
|---|---|---|---|
| **BadNets** <sub>Gu+ '17</sub> | 3×3 patch | dirty | The baseline every defense claims to beat |
| **Blended** <sub>Chen+ '17</sub> | global blend, α≈0.15 | dirty | No sparse mask exists to reconstruct |
| **SIG** <sub>Barni+ '19</sub> | sinusoidal overlay | **clean** | Labels are all correct — breaks data inspection |
| **WaNet** <sub>Nguyen+ '21</sub> | elastic warp | dirty | The reconstruction-killer. Noise mode also defeats STRIP |
| **Label-Consistent** <sub>Turner+ '19</sub> | patch + PGD perturbation | **clean** | Poisoned samples look correct *and* natural |
| **Adaptive-Blend** <sub>Qi+ '23</sub> | separability-suppressing | dirty | Purpose-built to evade latent-statistics defenses |

Poison rates are swept over ε ∈ {0.5%, 1%, 5%, 10%}, and BadNets is additionally run in the **all-to-all** label mapping — the case trigger reconstruction is structurally unable to detect, since its outlier test assumes exactly one class is unusually easy to reach and under all-to-all every class is. `badnets` and `badnets/all2all` are never averaged into one row.

Three of these apply a **weaker transform during training than at attack time**, and that asymmetry is the evasion mechanism rather than an implementation detail. So a trigger has three application paths, not one:

- `apply` — what the adversary sends at inference. ASR is measured against this.
- `apply_train` — what goes into the poisoned dataset. Adaptive-Blend blends only a random half of the image's tiles here, so poisoned samples never cluster.
- `apply_cover` — what goes on samples whose labels are left *correct*. WaNet's noise mode uses a different random warp, teaching the network that "warped" alone does not mean the target class.

Collapsing these into one method is the single most common way a reimplementation accidentally turns WaNet into something STRIP can catch.

Trigger patterns are generated from a recorded seed rather than shipped as image files. The published implementations use a copyrighted photograph, which makes the exact trigger unreproducible for anyone who does not have that file.

## Defenses

| Defense | Sees | Reports |
|---|---|---|
| **Neural Cleanse** <sub>Wang+ '19</sub> | weights + ~500 clean images | verdict, target label, reconstructed mask |
| **K-Arm** <sub>Shen+ '21</sub> | weights + ~500 clean images | same, with the budget spent adaptively |
| **STRIP** <sub>Gao+ '19</sub> | runtime inputs | per-input suspicion only |
| **Spectral Signatures** <sub>Tran+ '18</sub> | the (poisoned) training set | per-sample ranking, suspect class |
| **SPECTRE** <sub>Hayase+ '21</sub> | the (poisoned) training set | same, via robust covariance + QUE |
| **Activation Clustering** <sub>Chen+ '18</sub> | the (poisoned) training set | per-sample ranking, suspect class |
| **Fine-Pruning** <sub>Liu+ '18</sub> | weights + clean images | *mitigation* — separate scoreboard |

Neural Cleanse and K-Arm drive the *same* `TriggerOptimizer` and differ only in how they distribute optimisation steps across labels — which is K-Arm's entire claim, and is unattributable if the two have separate implementations. A test asserts they share the class. deadbolt runs them at **matched total budget** (K-Arm's `budget` = Neural Cleanse's `steps` × class count), so the comparison isolates the scheduler rather than measuring who was given more compute.

Spectral Signatures and SPECTRE are in the table together for the same reason: SPECTRE is the strongest member of the family whose central assumption Adaptive-Blend was built to falsify, and "does a better estimator rescue the assumption" is a question no single-paper evaluation can ask.

STRIP declares `produces_model_verdict = False`. It is an input filter and makes no claim about the weights; run it on a model whose attacker happens to be idle and it reports nothing, because there is nothing there to report. The harness scores it on per-input separation instead of inventing a model-level number for it.

Fine-Pruning is a mitigation, not a detector, so it returns a `MitigationResult` and lives in its own table. A method that drops ASR from 99% to 3% at the price of 15 points of clean accuracy has not defended anything — it has broken the model — and only reporting both numbers together makes that visible.

## Honesty guardrails

Benchmarks are easy to accidentally rig. These are the countermeasures:

- **Attack quality is a precondition, not a result.** An attack with ASR < 90% or a large clean-accuracy drop is not a valid test case, and is filtered out *before* defenses are scored.
- **Filtered runs stay in the record** with a reason field, and are counted in the report. Silently dropping cases is how benchmarks lie.
- **Calibration and evaluation splits are disjoint**, over models, shared across defenses.
- **Cover samples count as poisoned.** They carry the trigger, so a data filter that misses them has missed poisoned data. Scoring them as clean would flatter every latent-statistics defense against exactly the attacks designed to beat it.
- **Target-class samples are excluded** from both the ASR view and dirty-label poison selection. A triggered image already labelled with the target teaches nothing and inflates the number by ~1/C.
- **A crashed scan is recorded, not skipped.** Crashing on the attacks a method finds hardest is a result about that method.
- **Every result row records** git commit, seed, config hash, and wall-clock time.
- **Results are append-only JSONL.** The aggregate table is always regenerated from raw records and never hand-edited.

## Reproducing published failures is the test suite

A subtly broken detector still produces plausible-looking numbers, so unit tests are not enough. deadbolt's real correctness check is that it must reproduce known outcomes — including the failures. All of these run in [`tests/test_reproduction.py`](tests/test_reproduction.py):

| Check | Expected | If it fails |
|---|---|---|
| Neural Cleanse names the BadNets target | correct label | the λ schedule is broken |
| ...reconstructs it tightly | L1 < 25 for a 9-pixel patch | the λ schedule is broken |
| ...recovers the actual trigger pixels | mask IoU > 0.15 | it is flagging for the wrong reason |
| Neural Cleanse on WaNet | **finds no small mask** | our warp is too strong to be WaNet |
| STRIP on BadNets | per-input AUC > 0.9 | entropy normalisation is wrong |
| STRIP on WaNet + noise mode | **≈ chance** | noise mode is not actually training |
| Spectral Signatures on BadNets | per-sample AUC > 0.75 | feature extraction is wrong |

Failing to reproduce one of these is a bug report against deadbolt, not a discovery.

## Quickstart

```bash
git clone https://github.com/SoumilBhandari/deadbolt
cd deadbolt
uv venv --python 3.13 && uv pip install -e ".[dev]"
```

```bash
# Check the machine, and measure it rather than assuming it.
deadbolt doctor

# Verify the whole pipeline in ~2 minutes before committing 45 to Tier A.
# Six models is not a result, and the report says so in its own body.
deadbolt zoo build --config configs/experiments/smoke_mnist.yaml

# Tier A (MNIST + SmallCNN) — the full cycle end to end.
deadbolt zoo build --config configs/experiments/tierA_mnist.yaml
deadbolt scan --zoo tierA --defense neural_cleanse,karm,strip,spectral,spectre,activation_clustering
deadbolt report --zoo tierA --show
deadbolt mitigate --zoo tierA
```

Every command is resumable and idempotent: rebuilding a built zoo does nothing, and re-scanning skips pairs already recorded. These runs take hours, and a pipeline that cannot be interrupted is a pipeline nobody re-runs after changing one thing.

**Storage.** Datasets, checkpoints, and the zoo all live under one configurable root:

```bash
export DEADBOLT_ROOT=/Volumes/external/deadbolt   # defaults to ./runs
```

## Hardware

Developed on an **Apple M4 Max** via the PyTorch MPS backend; everything is sized to run on a laptop. `deadbolt doctor` benchmarks both the accelerator and the CPU rather than assuming the accelerator wins — for small models it sometimes does not.

<sub>MPS notes: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set, statistics run on CPU in float64 (MPS has no float64 and raises rather than falling back), and MPS is not bit-deterministic — results are reported over ≥3 seeds rather than claiming single-run reproducibility.</sub>

## Roadmap

| | Milestone | Gate | Status |
|---|---|---|---|
| ✅ | **M0** Environment, skeleton, core abstractions | MNIST CNN trains >98% on MPS | ✅ 99.46% over 51 seeds |
| ✅ | **M1** BadNets/Blended/SIG + zoo builder (Tier A) | ASR >95% at <1% clean-accuracy drop | dirty-label ✅ (ASR 0.9965); clean-label ❌ on MNIST |
| ✅ | **M2** Neural Cleanse, STRIP, Spectral, AC + metrics | NC names the right label; doesn't flag clean models | label ✅ 100%, clean models ❌ 43% FPR |
| ✅ | **M4** WaNet, Label-Consistent, all-to-all | *NC fails* — confirming this validates the harness | confirmed: 0.357 / 0.274 AUC |
| ✅ | **M5** K-Arm vs. linear scan | K-Arm matches NC's AUC at matched budget | matches: 0.924 vs 0.926 |
| ✅ | **M8** Mitigation: fine-pruning | ASR <10% at <2% clean-accuracy cost | ❌ gate missed: 64% ASR remains |
| ⬜ | **M3** Tier B: CIFAR-10 / PreAct ResNet-18 | ≥92% clean baseline; Tier A conclusions hold | config ready |
| 🔶 | **M6** SPECTRE, Input-aware attack, MNTD | — | SPECTRE done |
| ⬜ | **M7** Adaptive attacks targeting *our* defenses | — | |
| ⬜ | **M9** GTSRB (43 classes), ABS, TABOR | — | |

## Scope and intent

deadbolt is **defensive security research**. It implants backdoors for one reason: you cannot measure a detector without something to detect. Everything here reimplements attacks that are already published, peer-reviewed, and public, at research scale on public datasets (MNIST, CIFAR-10, GTSRB).

## References

Attacks — BadNets (Gu et al., 2017) · Blended (Chen et al., 2017) · Label-Consistent (Turner et al., 2019) · SIG (Barni et al., 2019) · WaNet (Nguyen & Tran, 2021) · Adaptive-Blend (Qi et al., 2023)

Defenses — Fine-Pruning (Liu et al., 2018) · Spectral Signatures (Tran et al., 2018) · Activation Clustering (Chen et al., 2018) · Neural Cleanse (Wang et al., 2019) · STRIP (Gao et al., 2019) · SPECTRE (Hayase et al., 2021) · K-Arm (Shen et al., 2021)

## License

MIT — see [LICENSE](LICENSE).

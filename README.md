<div align="center">

# 🔒 deadbolt

**A benchmark for catching backdoors in neural networks — and for finding out which detectors actually work.**

[![License: MIT](https://img.shields.io/badge/License-MIT-black.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/python-3.13-black.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-black.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/status-M0%20scaffolding-orange.svg)](#roadmap)

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

Each is typically evaluated by its own authors, against the attacks that existed when it was written, at a threshold tuned in the same paper, on metrics that vary between papers, with runtime costs frequently unreported. Defenses that look comparable on paper turn out to differ by orders of magnitude in cost and to fail on completely different inputs.

**deadbolt is the referee.** It implants a wide range of known backdoors into a large zoo of models, runs every detector against that zoo *blind*, and scores them all on identical metrics — including the ones they lose.

## What makes this a benchmark and not a demo

Three commitments, which are where most reimplementations quietly go wrong:

**1. Clean models are first-class citizens.**
A detector that flags everything has a perfect detection rate. False-positive rate is meaningless without a population of *benign* models to measure it against, so roughly half of deadbolt's zoo is clean models spanning multiple seeds and architectures.

**2. Detectors run blind.**
The harness hands a detector exactly `(checkpoint, small_clean_dataset)`. Ground truth — was this poisoned, with what, targeting which label — lives in a manifest the detector cannot read. Scoring happens outside the detector's reach.

**3. Negative results are the product.**
We report where defenses fail as loudly as where they succeed, because that is the information practitioners actually lack. Neural Cleanse is excellent against patch triggers and structurally cannot catch an all-to-all label mapping. That is not a footnote. That is the finding.

## The expected headline

Backdoor defenses do not fail randomly. They fail *structurally*, and the structure is predictable:

| Defense family | Sees | Strong against | Blind to |
|---|---|---|---|
| **Trigger reconstruction**<br><sub>Neural Cleanse, TABOR, K-Arm, ABS</sub> | Weights + ~500 clean images | Small, sparse, universal patch triggers | Warping (WaNet), global blends, all-to-all mappings, per-sample triggers |
| **Latent statistics**<br><sub>Spectral Signatures, Activation Clustering, SPECTRE</sub> | The (poisoned) training set | Dirty-label poisoning that separates in feature space | Clean-label attacks, low poison rates, attacks that deliberately suppress separability |
| **Input filtering**<br><sub>STRIP</sub> | Runtime inputs | Dominant patch triggers | WaNet's noise mode, subtle blends — and it never tells you the *model* is poisoned |
| **Meta-classification**<br><sub>MNTD</sub> | The weights, nothing else | The only option when you have no data at all | Expensive: needs hundreds of shadow models |

The interesting question is not "which defense is best." It is **"what does an attacker have to do to walk past all of them at once?"** — which is why deadbolt also implements adaptive attacks built specifically to evade the defenses in this repo.

## Design

```
                    ┌─────────────┐
   configs/  ─────► │  zoo build  │ ─────►  model zoo + manifest
   (yaml)           └─────────────┘         (~50% clean, ~50% poisoned)
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
    is_backdoored: bool          # verdict at the method's own published threshold
    score: float                 # continuous suspicion score → lets us plot ROC
    target_label: int | None     # did it name the right class?
    recovered_mask: Tensor | None  # scored by IoU against ground truth
    per_sample_scores: Tensor | None
    runtime_s: float             # reported in the main table, not a footnote
    extra: dict
```

Recording a continuous `score` *alongside* each paper's binary verdict is deliberate. It means we can compare methods on ROC curves rather than trusting thresholds that were tuned on the same data that produced the paper's numbers.

## Attacks

| Attack | Trigger | Label | Why it's here |
|---|---|---|---|
| **BadNets** <sub>Gu+ '17</sub> | 3×3 patch | dirty | The baseline every defense claims to beat |
| **Blended** <sub>Chen+ '17</sub> | global blend, α≈0.1 | dirty | No sparse mask exists to reconstruct |
| **SIG** <sub>Barni+ '19</sub> | sinusoidal overlay | **clean** | Labels are all correct — breaks data inspection |
| **WaNet** <sub>Nguyen+ '21</sub> | elastic warp | dirty | The reconstruction-killer. Noise mode also defeats STRIP |
| **Label-Consistent** <sub>Turner+ '19</sub> | patch + adversarial perturbation | **clean** | Poisoned samples look correct *and* natural |
| **Input-aware** <sub>Nguyen+ '20</sub> | per-sample | dirty | Breaks the universal-trigger assumption entirely |
| **Adaptive-Blend** <sub>Qi+ '23</sub> | separability-suppressing | dirty | Purpose-built to evade latent-statistics defenses |

Each runs in both **all-to-one** and **all-to-all** label mappings, swept over poison rates ε ∈ {0.5%, 1%, 5%, 10%}.

## Honesty guardrails

Benchmarks are easy to accidentally rig. These are the countermeasures:

- **Attack quality is a precondition, not a result.** An attack with ASR < 90% or a large clean-accuracy drop is not a valid test case. Weak attacks are filtered out *before* defenses are scored — otherwise every defense looks better than it is.
- **Filtered runs stay in the record** with a reason field. Silently dropping cases is how benchmarks lie.
- **Calibration and evaluation splits are disjoint.** Thresholds are tuned on a calibration slice of the zoo and never on the eval slice.
- **Every result row records** git commit, seed, config hash, and wall-clock time.
- **Results are append-only JSONL.** The aggregate table is always regenerated from raw records and never hand-edited.

## Reproducing published failures is the test suite

A subtly broken detector still produces plausible-looking numbers, so unit tests are not enough. deadbolt's real correctness check is that it must reproduce known, published outcomes:

| Check | Expected | If it fails |
|---|---|---|
| Neural Cleanse on BadNets | flags, names correct target label | λ schedule is likely broken |
| Neural Cleanse on clean models | flags ≲5% | λ schedule is broken |
| Neural Cleanse on WaNet | **fails to flag** | the WaNet warp is too strong — our bug |
| STRIP on BadNets | AUC > 0.95 | entropy normalization is wrong |
| STRIP on WaNet + noise mode | ≈ chance | noise mode isn't actually training |
| Spectral Signatures on Label-Consistent | **collapses** | clean-label poisoning isn't clean-label |

Failing to reproduce one of these is a bug report against deadbolt, not a discovery.

## Quickstart

> **Status:** M0 — scaffolding. The pipeline below is the target interface; see the [roadmap](#roadmap) for what actually runs today.

```bash
git clone https://github.com/SoumilBhandari/deadbolt
cd deadbolt
python3.13 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

```bash
# Tier A (MNIST + small CNN) — the full cycle, minutes not hours.
deadbolt zoo build --config configs/experiments/tierA_mnist.yaml
deadbolt scan --zoo tierA --defense neural_cleanse,strip,spectral,ac
deadbolt report --zoo tierA --out results/tierA
```

Tier A is deliberately small enough to run end-to-end in under an hour, which makes it a genuine regression test rather than an aspiration.

**Storage.** Datasets, checkpoints, and the zoo all live under one configurable root. Nothing is hardcoded:

```bash
export DEADBOLT_ROOT=/Volumes/external/deadbolt   # defaults to ./runs
```

## Roadmap

| | Milestone | Gate |
|---|---|---|
| ✅ | **M0** Environment, skeleton, core abstractions | MNIST CNN trains >98% on MPS |
| ⬜ | **M1** BadNets/Blended/SIG + zoo builder (Tier A) | ASR >95% at <1% clean-accuracy drop |
| ⬜ | **M2** Neural Cleanse, STRIP, Spectral Signatures, AC + metrics | NC names the right label; doesn't flag clean models |
| ⬜ | **M3** Scale to CIFAR-10 / PreAct ResNet-18 (Tier B) | ≥92% clean baseline; Tier A conclusions hold |
| ⬜ | **M4** WaNet, Label-Consistent, all-to-all | *NC fails* — confirming this validates the harness |
| ⬜ | **M5** GTSRB (43 classes), K-Arm vs. linear scan, ABS | K-Arm matches NC's AUC at lower cost |
| ⬜ | **M6** SPECTRE, Input-aware attack, MNTD | — |
| ⬜ | **M7** Adaptive attacks targeting *our* defenses | — |
| ⬜ | **M8** Mitigation: fine-pruning, trigger unlearning | ASR <10% at <2% clean-accuracy cost |
| ⬜ | **M9** Full matrix, ROC curves, cost table, writeup | — |

M0–M4 are the spine and produce a complete result on their own. M5–M8 are independent and reorderable.

## Hardware

Developed on an **Apple M1 Pro (16 GB)** via the PyTorch MPS backend — everything is sized to run on a laptop. Device is a config value (`mps` / `cuda` / `cpu`), so bursting to a rented GPU for the larger sweeps needs no code changes.

<sub>MPS notes: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set, statistics run on CPU in float64, and MPS is not bit-deterministic — results are reported over ≥3 seeds rather than claiming single-run reproducibility.</sub>

## Scope and intent

deadbolt is **defensive security research**. It implants backdoors for one reason: you cannot measure a detector without something to detect. Everything here reimplements attacks that are already published, peer-reviewed, and public, at research scale on public datasets (MNIST, CIFAR-10, GTSRB).

## References

Attacks — BadNets (Gu et al., 2017) · Blended (Chen et al., 2017) · Label-Consistent (Turner et al., 2019) · SIG (Barni et al., 2019) · Input-aware (Nguyen & Tran, 2020) · WaNet (Nguyen & Tran, 2021) · Adaptive-Blend (Qi et al., 2023)

Defenses — Fine-Pruning (Liu et al., 2018) · Spectral Signatures (Tran et al., 2018) · Activation Clustering (Chen et al., 2018) · Neural Cleanse (Wang et al., 2019) · TABOR (Guo et al., 2019) · ABS (Liu et al., 2019) · STRIP (Gao et al., 2019) · SPECTRE (Hayase et al., 2021) · MNTD (Xu et al., 2021) · K-Arm (Shen et al., 2021)

## License

MIT — see [LICENSE](LICENSE).

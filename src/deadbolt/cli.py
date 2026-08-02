"""Command-line interface.

Five verbs, matching the five things you actually do with a benchmark:

``doctor``   check the machine can run it, and how fast
``zoo``      build (or extend) a population of models
``scan``     run detectors against a zoo, blind
``report``   regenerate the results table from raw records
``mitigate`` try to repair backdoored models and measure what it cost

Every command is resumable and every command is idempotent. Rebuilding a zoo
that is already built does nothing; re-scanning skips pairs already recorded.
This is not politeness — these runs take hours, and a pipeline that cannot be
interrupted is a pipeline nobody re-runs after changing one thing.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from deadbolt import __version__


def _fmt_secs(s: float) -> str:
    if s < 90:
        return f"{s:.0f}s"
    if s < 5400:
        return f"{s / 60:.1f}m"
    return f"{s / 3600:.1f}h"


def cmd_doctor(args: argparse.Namespace) -> int:
    """Report the environment, and measure it rather than assuming it."""
    from deadbolt.checkpoints import build_model, roundtrip_error
    from deadbolt.config import resolve_device, storage_root

    print(f"deadbolt {__version__}")
    print(f"python   {sys.version.split()[0]}")
    print(f"torch    {torch.__version__}")
    print(f"root     {storage_root()}")
    print(
        "devices  "
        + ", ".join(
            n
            for n, ok in (
                ("mps", torch.backends.mps.is_available()),
                ("cuda", torch.cuda.is_available()),
                ("cpu", True),
            )
            if ok
        )
    )

    device = resolve_device(args.device)
    print(f"selected {device}")

    # Benchmark both devices rather than trusting that the accelerator wins.
    # For Tier A models it frequently does not: the kernels are small enough
    # that launch overhead dominates, and a sweep that picks MPS on faith can
    # be slower than one that measures.
    from deadbolt.data.datasets import SPECS

    spec = SPECS[args.dataset]
    model = build_model(args.arch, args.dataset, args.width)
    x = torch.rand(args.batch_size, spec.in_channels, *spec.image_size)
    for dev in {str(device), "cpu"}:
        try:
            m, xx = model.to(dev), x.to(dev)
            for _ in range(3):
                m(xx)
            if dev == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                m(xx)
            if dev == "mps":
                torch.mps.synchronize()
            dt = (time.perf_counter() - t0) / 20
            print(f"  {dev:5s} {args.arch} fwd batch={args.batch_size}: {dt * 1000:.1f} ms")
        except Exception as exc:  # noqa: BLE001 - diagnostics must not crash
            print(f"  {dev:5s} unavailable: {exc}")

    err = roundtrip_error(model.cpu(), x.cpu())
    print(
        f"fp16 checkpoint cost: max logit shift {err['max_logit_shift']:.2e}, "
        f"prediction flips {err['pred_flip_rate']:.4f}"
    )
    return 0


def cmd_zoo(args: argparse.Namespace) -> int:
    from deadbolt.zoo import ZooSpec, build, load_manifest, summarise

    spec = ZooSpec.from_yaml(args.config)
    if args.name:
        spec.name = args.name
    planned = spec.expand()
    print(f"zoo '{spec.name}': {len(planned)} models planned")
    if args.dry_run:
        for cfg in planned:
            print(f"  {cfg.attack or 'clean':16s} seed={cfg.seed} rate={cfg.poison_rate}")
        return 0

    started = time.perf_counter()
    state = {"n": 0}

    def progress(i: int, total: int, cfg, result) -> None:
        state["n"] += 1
        elapsed = time.perf_counter() - started
        remaining = (total - i - 1) * elapsed / state["n"]
        asr = "" if result.attack_success_rate is None else f" asr={result.attack_success_rate:.4f}"
        flag = "" if result.valid_testcase else f"  FILTERED: {result.filter_reason}"
        # A clean model has no poison rate; printing the TrainConfig default
        # here would suggest 5% of the benign models were poisoned.
        rate = f"{cfg.poison_rate:<6}" if cfg.attack else "—     "
        print(
            f"[{i + 1:>4}/{total}] {cfg.attack or 'clean':16s} seed={cfg.seed} "
            f"rate={rate} clean={result.clean_accuracy:.4f}{asr} "
            f"({_fmt_secs(result.train_seconds)}, eta {_fmt_secs(remaining)}){flag}",
            flush=True,
        )

    build(spec, resume=not args.no_resume, on_model=progress, limit=args.limit)
    s = summarise(load_manifest(spec.name))
    print(
        f"\n{s['n_models']} models: {s['n_clean']} clean, {s['n_poisoned']} poisoned "
        f"({s['n_valid_poisoned']} valid). clean acc {s['mean_clean_accuracy']:.4f}, "
        f"mean ASR {s['mean_asr']:.4f}"
    )
    for f in s["filtered"]:
        print(f"  filtered: {f['checkpoint']}  {f['reason']}")
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    from deadbolt.defenses import DEFENSES
    from deadbolt.scan import scan_zoo
    from deadbolt.zoo import load_manifest, scannable

    names = [n.strip() for n in args.defense.split(",") if n.strip()]
    unknown = [n for n in names if n not in DEFENSES]
    if unknown:
        print(f"unknown defenses: {unknown}; known: {sorted(DEFENSES)}", file=sys.stderr)
        return 2

    overrides = json.loads(args.defense_args) if args.defense_args else {}
    detectors = [DEFENSES[n](**overrides.get(n, {})) for n in names]

    manifest = load_manifest(args.zoo)
    if not manifest:
        print(f"zoo '{args.zoo}' has no manifest; run `deadbolt zoo build` first", file=sys.stderr)
        return 2

    records = list(scannable(manifest))
    if args.limit:
        records = records[: args.limit]
    total = len(records) * len(detectors)
    print(f"scanning {len(records)} models x {len(detectors)} defenses = {total} runs")

    started = time.perf_counter()
    state = {"n": 0}

    def progress(rec) -> None:
        state["n"] += 1
        elapsed = time.perf_counter() - started
        eta = (total - state["n"]) * elapsed / state["n"]
        if rec.error:
            print(f"[{state['n']:>5}/{total}] {rec.defense:22s} ERROR {rec.error}", flush=True)
            return
        truth = "poisoned" if rec.truth["is_backdoored"] else "clean   "
        print(
            f"[{state['n']:>5}/{total}] {rec.defense:22s} {truth} "
            f"score={rec.result['score']:8.3f} verdict={str(rec.result['is_backdoored']):5s} "
            f"({rec.result['runtime_s']:.1f}s, eta {_fmt_secs(eta)})",
            flush=True,
        )

    scan_zoo(
        args.zoo,
        records,
        detectors,
        on_scan=progress,
        resume=not args.no_resume,
        trainset_size=args.trainset_size,
    )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from deadbolt.report import roc_curves, write_report
    from deadbolt.scan import load_scans
    from deadbolt.zoo import load_manifest

    scans = load_scans(args.zoo)
    if not scans:
        print(f"zoo '{args.zoo}' has no scans; run `deadbolt scan` first", file=sys.stderr)
        return 2
    out = Path(args.out or f"results/{args.zoo}")
    path = write_report(args.zoo, out, scans, load_manifest(args.zoo))
    fig = roc_curves(scans, out / "figures" / "roc.png")
    print(f"wrote {path}")
    print(f"wrote {out / 'aggregate.json'}")
    if fig:
        print(f"wrote {fig}")
    if args.show:
        print()
        print(path.read_text())
    return 0


def cmd_mitigate(args: argparse.Namespace) -> int:
    import json as _json

    from torch.utils.data import DataLoader

    from deadbolt.checkpoints import load_model
    from deadbolt.config import resolve_device, zoo_dir
    from deadbolt.data.datasets import load_clean, subset
    from deadbolt.data.poison import make_asr_view
    from deadbolt.data.splits import defender_split
    from deadbolt.mitigate import fine_prune
    from deadbolt.train import TrainConfig, build_trigger, evaluate
    from deadbolt.zoo import load_manifest, scannable

    device = resolve_device(args.device)
    records = [r for r in scannable(load_manifest(args.zoo)) if r.is_backdoored]
    if args.limit:
        records = records[: args.limit]
    print(f"mitigating {len(records)} backdoored models with fine-pruning")

    out_path = zoo_dir(args.zoo) / "mitigations.jsonl"
    for i, record in enumerate(records):
        cfg = TrainConfig(**record.config)
        trigger = build_trigger(cfg)
        test, labels = load_clean(cfg.dataset, train=False)
        d_idx, eval_idx = defender_split(labels, cfg.defender_per_class, cfg.seed)
        clean_loader = DataLoader(subset(test, d_idx), batch_size=128, shuffle=True)
        eval_set = subset(test, eval_idx)
        eval_loader = DataLoader(eval_set, batch_size=512)
        asr_loader = DataLoader(make_asr_view(eval_set, trigger, labels[eval_idx]), batch_size=512)

        def eval_fn(model, clean_only: bool = False):
            acc = evaluate(model, eval_loader, device)
            return acc, (None if clean_only else evaluate(model, asr_loader, device))

        model = load_model(record.checkpoint, device)
        result = fine_prune(
            model, clean_loader, eval_fn, device, prune_fraction=args.prune_fraction
        )
        row = {"checkpoint": record.checkpoint, "attack": record.poison["attack"], **result.as_record()}
        with out_path.open("a") as f:
            f.write(_json.dumps(row) + "\n")
        print(
            f"[{i + 1}/{len(records)}] {record.poison['attack']:16s} "
            f"asr {result.asr_before:.3f} -> {result.asr_after:.3f}  "
            f"clean {result.clean_accuracy_before:.3f} -> {result.clean_accuracy_after:.3f}  "
            f"({result.extra['pruned_channels']}/{result.extra['total_channels']} channels)",
            flush=True,
        )
    print(f"wrote {out_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deadbolt", description=__doc__.split("\n")[0])
    p.add_argument("--version", action="version", version=f"deadbolt {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("doctor", help="check the environment and benchmark the device")
    d.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    d.add_argument("--arch", default="smallcnn")
    d.add_argument("--dataset", default="cifar10")
    d.add_argument("--width", type=int, default=32)
    d.add_argument("--batch-size", type=int, default=128)
    d.set_defaults(func=cmd_doctor)

    z = sub.add_parser("zoo", help="build a model zoo")
    zsub = z.add_subparsers(dest="zoo_command", required=True)
    zb = zsub.add_parser("build", help="train every model in a config")
    zb.add_argument("--config", required=True, help="path to a zoo YAML")
    zb.add_argument("--name", help="override the zoo name in the config")
    zb.add_argument("--limit", type=int, help="stop after this many new models")
    zb.add_argument("--dry-run", action="store_true", help="list planned models and exit")
    zb.add_argument("--no-resume", action="store_true", help="retrain models already in the manifest")
    zb.set_defaults(func=cmd_zoo)

    s = sub.add_parser("scan", help="run detectors against a zoo, blind")
    s.add_argument("--zoo", required=True)
    s.add_argument("--defense", required=True, help="comma-separated detector names")
    s.add_argument("--defense-args", help="JSON: {\"neural_cleanse\": {\"steps\": 200}}")
    s.add_argument("--limit", type=int, help="scan only the first N models")
    s.add_argument("--trainset-size", type=int, default=10_000)
    s.add_argument("--no-resume", action="store_true")
    s.set_defaults(func=cmd_scan)

    r = sub.add_parser("report", help="regenerate the results table from raw records")
    r.add_argument("--zoo", required=True)
    r.add_argument("--out", help="output directory (default results/<zoo>)")
    r.add_argument("--show", action="store_true", help="print the report to stdout")
    r.set_defaults(func=cmd_report)

    m = sub.add_parser("mitigate", help="repair backdoored models and measure the cost")
    m.add_argument("--zoo", required=True)
    m.add_argument("--device", default="auto", choices=["auto", "mps", "cuda", "cpu"])
    m.add_argument("--prune-fraction", type=float, default=0.6)
    m.add_argument("--limit", type=int)
    m.set_defaults(func=cmd_mitigate)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

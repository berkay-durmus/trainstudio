"""End-to-end: scan -> recommend -> config.json -> runner.py -> checkpoint -> predict."""
import json, os, subprocess, sys, time
from pathlib import Path
sys.path.insert(0, os.getcwd())

from core.recommend import recommend
from core.registry import get
from core.schemas import DatasetConfig, ModelSelection, RunConfig, Task
from core.hardware import detect
from data.scan import scan_dataset

# Defaults to <E2E>/data; fixtures.py points it straight at the generated
# datasets so they are not copied a second time.
DATA = Path(os.environ.get("TS_UI_DATA") or Path(os.environ["E2E"]) / "data")
OUT  = Path(os.environ["E2E"]) / "runs"

def make_ds(root, task):
    res = scan_dataset(root, task=task)
    assert res.task is not None, f"scan failed: {[i.title for i in res.issues]}"
    return DatasetConfig(
        root=str(root), task=res.task, modality=res.modality, classes=res.classes,
        n_train=res.n_train, n_val=res.n_val, n_test=res.n_test,
        class_counts=(res.totals_per_class() if res.class_counts else {}),
        median_image_size=res.median_size, channels=3,
        splits_file=getattr(res, "splits_file", None),
    ), res

def run(spec_id, root, task, name):
    spec = get(spec_id); assert spec, spec_id
    ds, res = make_ds(root, task)
    print(f"\n=== {name}: {spec.display_name} ({spec.backend.value}) ===")
    print(f"    scan: task={res.task.value} classes={res.classes} "
          f"train/val/test={res.n_train}/{res.n_val}/{res.n_test}")

    rec = recommend(spec, ds, detect())
    hp = rec.hp if hasattr(rec, "hp") else rec
    # shrink to something that finishes on a laptop CPU
    hp.epochs, hp.batch_size, hp.img_size = 2, 4, 64
    hp.num_workers, hp.amp, hp.channels_last = 0, False, False
    hp.early_stopping, hp.preview_every_n_epochs = False, 1
    hp.pretrained = False          # no network dependency in the test

    cfg = RunConfig(
        run_name=name, output_dir=str(OUT), created_at=time.strftime("%F %T"),
        dataset=ds,
        model=ModelSelection(
            spec_id=spec.id, backend=spec.backend, arch=spec.arch,
            display_name=spec.display_name,
            encoder=hp.encoder if spec.needs_encoder else None,
            encoder_weights="imagenet" if spec.needs_encoder and hp.pretrained else None,
            weights=spec.weights,
        ),
        hp=hp, aug=rec.aug if hasattr(rec, "aug") else None, device="cpu",
    )
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = cfg.save()
    print(f"    config: {cfg_path}")

    r = subprocess.run([sys.executable, "runner.py", "--config", str(cfg_path)],
                       capture_output=True, text=True)
    print(f"    runner exit={r.returncode}")
    if r.returncode != 0:
        print("    ---- stdout ----"); print(r.stdout[-3000:])
        print("    ---- stderr ----"); print(r.stderr[-3000:])
        return False

    best = cfg.run_dir / "checkpoints" / "best.pt"
    alt  = list(cfg.run_dir.rglob("best.pt"))
    ck = best if best.exists() else (alt[0] if alt else None)
    print(f"    checkpoint: {ck} ({ck.stat().st_size//1024 if ck else 0} KB)")
    if ck is None:
        print("    files:", [str(p.relative_to(cfg.run_dir)) for p in cfg.run_dir.rglob('*') if p.is_file()][:20])
        return False

    # metrics
    mfile = cfg.run_dir / "metrics.json"
    if mfile.exists():
        print("    metrics.json keys:", list(json.loads(mfile.read_text()))[:8])

    # inference round-trip
    from export.inference import load_checkpoint, predict
    import numpy as np
    loaded = load_checkpoint(cfg.run_dir)
    print(f"    loaded: backend={loaded.backend.value} classes={loaded.classes} epoch={loaded.epoch}")
    img = np.random.randint(0, 255, (96, 96, 3), dtype=np.uint8)
    p = predict(loaded, img)
    if p.task == Task.CLASSIFICATION:
        print(f"    predict: label={p.label} conf={p.confidence:.3f} probs={p.probs.shape} "
              f"({p.inference_ms:.0f} ms)")
    else:
        print(f"    predict: mask={p.mask.shape} classes={sorted(set(p.mask.flatten().tolist()))} "
              f"areas={ {k: round(v,3) for k,v in p.class_areas.items()} } ({p.inference_ms:.0f} ms)")
    return True

ok = []
ok.append(run("tv_resnet18", DATA / "cls_shapes", Task.CLASSIFICATION, "tv-cls"))
ok.append(run("tv_lraspp_mobilenet_v3_large", DATA / "seg_shapes", Task.SEGMENTATION, "tv-seg"))
print("\n==== RESULT:", "ALL PASSED" if all(ok) else "FAILURES PRESENT", "====")
sys.exit(0 if all(ok) else 1)

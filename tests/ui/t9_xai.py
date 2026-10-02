"""UI test 9 — explaining predictions: Grad-CAM, LIME and SHAP.

The fixture runs are explained for real; the architectures the fixtures do not
cover are built with random weights, which is enough to find Grad-CAM's layer
and check the shape of what comes out. One deletion test checks that the maps
mean something: blurring the region a method ranks highest must lower the
prediction far more than blurring a random region of the same size.
"""
import os, sys, tempfile, time, warnings

TMP = os.path.realpath(tempfile.mkdtemp(prefix="ts_t9_"))
os.environ["TRAINSTUDIO_HOME"] = os.path.join(TMP, "home")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import T, PROJ, record, run_page, summary

import cv2
import numpy as np
import torch
import ui.state as K
from core import launcher, runs
from core.schemas import Backend, Layout, RunConfig, RunStatus, Task
from export import explain as X
from export.inference import LoadedModel, _rebuild, load_checkpoint, predict, read_input

warnings.filterwarnings("ignore")
RUNS = os.path.join(T, "e2e", "runs")
DATA = os.path.join(T, "ts_data")


def sample(sub, n=1):
    base = os.path.join(DATA, sub)
    files = sorted(os.path.join(r, f) for r, _, fs in os.walk(base) for f in fs
                   if f.endswith((".png", ".jpg")))
    return files[:n]


def target_of(L, img):
    p = predict(L, img)
    if p.probs is not None:
        return int(p.probs.argmax())
    counts = np.bincount(p.mask.ravel(), minlength=len(L.classes))
    return int(counts[1:].argmax()) + 1 if counts[1:].any() else 0


# ══ 1 · Every method on the fixture runs ═══════════════════════════════════
print("\n1 · Grad-CAM, LIME and SHAP on the fixture runs")
for run, sub in (("tv-cls", "cls_shapes/test"), ("tv-seg", "seg_shapes/test/images")):
    L = load_checkpoint(os.path.join(RUNS, run))
    img = read_input(sample(sub)[0], L)
    t = target_of(L, img)
    av = X.available(L)
    record(run, "every method is available", all(ok for ok, _ in av.values()), str(av))
    for m in X.METHODS:
        e = X.explain(L, img, m, t, "fast")
        record(run, f"{m}: an overlay the size of the input, a finite map",
               e.overlay.shape == (*img.shape[:2], 3) and e.overlay.dtype == np.uint8
               and np.isfinite(e.heat).all() and e.heat.shape == img.shape[:2] and e.elapsed_s > 0,
               f"overlay={e.overlay.shape} heat={e.heat.shape} {e.elapsed_s:.2f}s")
    for m in ("lime", "shap"):
        a = X.explain(L, img, m, t, "fast").heat
        b = X.explain(L, img, m, t, "fast").heat
        record(run, f"{m}: the same input gives the same map", np.allclose(a, b, atol=1e-6))
    other = (t + 1) % len(L.classes)
    a = X.explain(L, img, "shap", t, "fast").heat
    b = X.explain(L, img, "shap", other, "fast").heat
    record(run, "another class gives another map", not np.allclose(a, b, atol=1e-3))

# ══ 2 · The maps mean something: deletion ══════════════════════════════════
print("\n2 · Blurring the top-ranked fifth lowers the prediction more than a random fifth")
L = load_checkpoint(os.path.join(RUNS, "tv-cls"))
m = X._Model(L)
rng = np.random.default_rng(0)


def prob(img, t):
    w = X._work_image(img, m.size)
    return float(X.ScoreFn(m, w, t)(w[None])[0, 0])


files = sample("cls_shapes/test", 8)
for meth, factor in (("gradcam", 1.5), ("lime", 3.0), ("shap", 3.0)):
    top, rnd = [], []
    for f in files:
        img = read_input(f, L)
        t = target_of(L, img)
        heat = X.explain(L, img, meth, t, "fast").heat.ravel()
        blur = cv2.GaussianBlur(img, (0, 0), 6).reshape(-1, 3)
        k = int(0.2 * heat.size)
        p0 = prob(img, t)
        for idx, acc in ((np.argsort(-heat)[:k], top), (rng.choice(heat.size, k, replace=False), rnd)):
            im2 = img.copy().reshape(-1, 3)
            im2[idx] = blur[idx]
            acc.append(p0 - prob(im2.reshape(img.shape), t))
    record("deletion", f"{meth}: the top fifth matters more than a random fifth",
           np.mean(top) > factor * max(np.mean(rnd), 0.005),
           f"drop {np.mean(top):.3f} vs random {np.mean(rnd):.3f}")

# ══ 3 · Grad-CAM's layer, across architectures ═════════════════════════════
print("\n3 · Grad-CAM's layer on ViT, Swin and CNNs (random weights)")
t = X.reshape_transform(torch.nn.Module())
record("shape", "ViT tokens fold to a grid, the class token dropped",
       tuple(t(torch.zeros(2, 197, 8)).shape) == (2, 8, 14, 14))
record("shape", "NHWC (Swin) is permuted to NCHW",
       tuple(t(torch.zeros(2, 7, 7, 16)).shape) == (2, 16, 7, 7))
record("shape", "NCHW is left alone", tuple(t(torch.zeros(2, 16, 7, 7)).shape) == (2, 16, 7, 7))

cls_cfg = RunConfig.load(os.path.join(RUNS, "tv-cls", Layout.CONFIG))
seg_cfg = RunConfig.load(os.path.join(RUNS, "tv-seg", Layout.CONFIG))
CASES = [
    (cls_cfg, Backend.TIMM, "vit_base_patch16_224.augreg2_in21k_ft_in1k", 224, None),
    (cls_cfg, Backend.TIMM, "swinv2_base_window8_256.ms_in1k", 256, None),
    (cls_cfg, Backend.TIMM, "convnextv2_tiny.fcmae_ft_in22k_in1k", 224, None),
    (cls_cfg, Backend.TIMM, "efficientnet_b0.ra_in1k", 224, None),
    (cls_cfg, Backend.TORCHVISION, "vit_b_16", 224, None),
    (cls_cfg, Backend.TORCHVISION, "swin_v2_t", 224, None),
    (seg_cfg, Backend.SMP, "FPN", 128, "resnet34"),
    (seg_cfg, Backend.HF, "nvidia/segformer-b0-finetuned-ade-512-512", 128, None),
]
torch.manual_seed(0)
for base, backend, arch, size, encoder in CASES:
    cfg = base.model_copy(deep=True)
    cfg.model.arch, cfg.model.backend, cfg.model.encoder = arch, backend, encoder
    cfg.hp.img_size = size
    try:
        L = LoadedModel(model=_rebuild(cfg).eval(), cfg=cfg, device=torch.device("cpu"),
                        classes=cfg.dataset.classes)
        img = np.random.default_rng(1).integers(0, 255, (size, size, 3), dtype=np.uint8)
        layer = X._target_layer(X._Model(L), X._work_image(img, size))
        best = max(X.explain(L, img, "gradcam", c).heat.std() for c in range(len(L.classes)))
        record("arch", f"{arch}: Grad-CAM finds a layer and maps it",
               best > 0.01, f"{type(layer).__name__}, std {best:.3f}")
    except Exception as exc:
        record("arch", f"{arch}: Grad-CAM finds a layer and maps it", False,
               f"{type(exc).__name__}: {exc}"[:200])

# ══ 4 · Ultralytics classification ═════════════════════════════════════════
print("\n4 · An Ultralytics classifier")
from core.recommend import recommend
from core.hardware import detect
from core.registry import get
from core.schemas import DatasetConfig, ModelSelection
from data.scan import scan_dataset

spec = get("yolo26n_cls")
root = os.path.join(DATA, "cls_shapes")
res = scan_dataset(root, task=Task.CLASSIFICATION)
ds = DatasetConfig(root=root, task=res.task, modality=res.modality, classes=res.classes,
                   n_train=res.n_train, n_val=res.n_val, n_test=res.n_test,
                   class_counts=res.totals_per_class(), median_image_size=res.median_size,
                   channels=3)
rec = recommend(spec, ds, detect())
hp = rec.hp
hp.epochs, hp.batch_size, hp.img_size, hp.num_workers = 1, 8, 64, 0
hp.pretrained, hp.amp = False, False
cfg = RunConfig(run_name="yolo", output_dir=os.path.join(TMP, "runs"), created_at=time.strftime("%F %T"),
                dataset=ds, model=ModelSelection(spec_id=spec.id, backend=spec.backend,
                                                 arch=spec.arch, display_name=spec.display_name),
                hp=hp, aug=rec.aug, device="cpu")
launcher.start_run(cfg)
t0 = time.time()
while time.time() - t0 < 600:
    s = runs.load_summary(cfg.run_dir)
    if s.status.is_terminal and not runs.process_alive(s.pid):
        break
    time.sleep(0.5)
record("yolo", "a one-epoch YOLO classifier trains", s.status == RunStatus.COMPLETED, s.error or "")
if s.status == RunStatus.COMPLETED:
    L = load_checkpoint(cfg.run_dir)
    img = read_input(sample("cls_shapes/test")[0], L)
    t = target_of(L, img)
    w = X._work_image(img, X._Model(L).size)
    ours = X.ScoreFn(X._Model(L), w, t)(w[None])[0, 0]
    record("yolo", "the explained score is Ultralytics' own probability",
           abs(ours - predict(L, img).probs[t]) < 0.02, f"{ours:.3f} vs {predict(L, img).probs[t]:.3f}")
    for meth in X.METHODS:
        try:
            e = X.explain(L, img, meth, t, "fast")
            ok, detail = e.overlay.shape[:2] == img.shape[:2] and np.isfinite(e.heat).all(), ""
        except Exception as exc:
            ok, detail = False, f"{type(exc).__name__}: {exc}"
        record("yolo", f"{meth} explains it", ok, detail)

# ══ 5 · A missing library ══════════════════════════════════════════════════
print("\n5 · Without shap installed")
saved = sys.modules.get("shap")
sys.modules["shap"] = None                  # makes `import shap` raise ImportError
L = load_checkpoint(os.path.join(RUNS, "tv-cls"))
ok, why = X.available(L)["shap"]
record("deps", "SHAP is reported unavailable, with the install command",
       not ok and "pip install shap" in why, why)
try:
    X.explain(L, read_input(sample("cls_shapes/test")[0], L), "shap", 0)
    raised = False
except X.ExplainError:
    raised = True
record("deps", "and explain() refuses it with that reason", raised)
if saved is not None:
    sys.modules["shap"] = saved
else:
    del sys.modules["shap"]

# ══ 6 · The Inference page ═════════════════════════════════════════════════
print("\n6 · The Inference page")


def captions(at):
    return [c for im in at.image for c in im.captions]


def open_panel(run):
    at = run_page("views/6_Inference.py",
                  lambda a: [a.session_state.__setitem__(k, v) for k, v in
                             {K.K_OUTPUT: RUNS, K.K_ACTIVE_RUN: os.path.join(RUNS, run)}.items()],
                  timeout=600)
    at = [r for r in at.radio if "source" in (r.label or "").lower()][0] \
        .set_value("Pick from the dataset").run()
    at = [t for t in at.toggle if "Explain this prediction" in (t.label or "")][0].set_value(True).run()
    return at


at = open_panel("tv-cls")
record("ui", "the panel opens on Grad-CAM, already drawn",
       not at.exception and any(c.startswith("Grad-CAM ·") for c in captions(at)),
       f"{captions(at)} {[e.value for e in at.exception][:1]}")
at = at.radio(key="xai_m_single").set_value("lime").run()
go = [b for b in at.button if (b.key or "") == "xai_go_single"]
record("ui", "LIME waits for its button, which says what it costs",
       bool(go) and "LIME" in go[0].label and "model evaluations" in " ".join(
           c.value for c in at.caption), f"{[b.label for b in at.button]}")
at = go[0].click().run() if go else at
record("ui", "LIME is drawn once run", any(c.startswith("LIME ·") for c in captions(at)),
       str(captions(at)))
at = at.toggle(key="xai_all_single").set_value(True).run()
go = [b for b in at.button if (b.key or "") == "xai_go_single"]
record("ui", "Compare keeps what already ran and offers only SHAP",
       bool(go) and go[0].label == "▶ Run SHAP", f"{[b.label for b in at.button]}")
at = go[0].click().run() if go else at
shown = [c for c in captions(at) if c.split(" ·")[0] in ("Grad-CAM", "LIME", "SHAP")]
record("ui", "Compare shows all three side by side", len(shown) == 3 and not at.exception,
       str(shown))

at = open_panel("tv-seg")
record("ui", "a segmentation prediction is explained too",
       not at.exception and any(c.startswith("Grad-CAM ·") for c in captions(at)),
       f"{captions(at)} {[e.value for e in at.exception][:1]}")

import shutil
shutil.rmtree(TMP, ignore_errors=True)
sys.exit(summary("UI TEST 9 · explanations"))

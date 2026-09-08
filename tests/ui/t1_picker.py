"""UI test 1 — the folder picker: every way a human can supply a path."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness import T, dp, record, run_page, blob, messages, summary, ss_get

PAGE = "pages/1_Dataset.py"
GOOD = os.path.join(T, "ts_data", "cls_shapes")          # a real dataset
TYPED = dp("dataset", "typed")

def type_path(raw, then_confirm=False):
    """Put `raw` in the picker's text box, rerun, optionally press Select."""
    at = run_page(PAGE)
    at.text_input(key=TYPED).set_value(raw).run()
    if then_confirm:
        for b in at.button:
            if b.proto.id.endswith("pick") or "Select this folder" in (b.label or ""):
                b.click().run(); break
    return at

# ── 1. Forms of a valid path that must all resolve ──────────────────────────
print("\n1 · Path forms that must be accepted")
variants = {
    "plain absolute":        GOOD,
    "trailing whitespace":   GOOD + "   ",
    "trailing newline":      GOOD + "\n",
    "leading whitespace":    "   " + GOOD,
    "double quoted":         f'"{GOOD}"',
    "single quoted":         f"'{GOOD}'",
    "file:// URI":           "file://" + GOOD,
    "trailing slash":        GOOD + "/",
    "dot segment":           os.path.join(os.path.dirname(GOOD), ".", "cls_shapes"),
    "parent traversal":      os.path.join(GOOD, "..", "cls_shapes"),
    "$VAR expansion":        "$TSROOT/ts_data/cls_shapes",
    "~ expansion":           None,      # filled in below when possible
}
os.environ["TSROOT"] = T

home = os.path.expanduser("~")
variants["~ expansion"] = "~" + GOOD[len(home):] if GOOD.startswith(home) else GOOD

for name, raw in variants.items():
    at = type_path(raw)
    errs = messages(at, "error")
    # accepted == the picker navigated there and offers it for selection
    cwd = ss_get(at, dp("dataset", "cwd"), "")
    ok = not errs and os.path.realpath(str(cwd)) == os.path.realpath(GOOD)
    record("paths", name, ok, "" if ok else f"cwd={cwd} errors={errs[:1]}")

# a percent-encoded path with a space, via the spacedir fixture
SPACED = os.path.join(T, "spacedir", "My Data Set")
for name, raw in {
    "percent-encoded space": "file://" + SPACED.replace(" ", "%20"),
    "shell-escaped space":   SPACED.replace(" ", "\\ "),
    "quoted with space":     f'"{SPACED}"',
}.items():
    at = type_path(raw)
    cwd = str(ss_get(at, dp("dataset", "cwd"), ""))
    ok = not messages(at, "error") and os.path.realpath(cwd) == os.path.realpath(SPACED)
    record("paths", name, ok, "" if ok else f"cwd={cwd}")

# ── 2. Bad paths must each give their own explanation ───────────────────────
print("\n2 · Rejected paths, each with a specific message")
cases = [
    ("nonexistent leaf", os.path.join(GOOD, "nope_missing"),
     ["there is no", "not found"]),
    ("nonexistent deep", "/definitely/not/here/at/all",
     ["not found", "deepest existing"]),
    ("a file, not a folder", os.path.join(T, "edge_afile.txt"),
     ["is a file, not a folder"]),
    ("empty input", "   ", []),
]
for name, raw, expect in cases:
    at = type_path(raw)
    msg = " ".join(messages(at, "error") + messages(at, "warning")).lower()
    if not expect:
        ok = True                       # empty simply must not crash
        detail = ""
    else:
        ok = any(e in msg for e in expect)
        detail = "" if ok else f"got: {msg[:120]!r}"
    record("paths", name, ok and not at.exception, detail)

# ── 3. The stale-selection trap ─────────────────────────────────────────────
print("\n3 · A failed path must not confirm the folder on screen")
at = run_page(PAGE)
start_cwd = str(ss_get(at, dp("dataset", "cwd"), ""))
at.text_input(key=TYPED).set_value("/definitely/not/here").run()
# the caption must name what would actually be selected
shown = blob(at)
sel_before = ss_get(at, dp("dataset", "selected"))
record("stale", "no selection is committed by a failed path", sel_before is None,
       f"selected={sel_before}")
record("stale", "the page states which folder is being selected",
       "selecting:" in shown, "")

# ── 4. Unreadable folders ───────────────────────────────────────────────────
print("\n4 · Permission-denied folders report, not crash")
for name, path, expect in [
    ("unreadable dataset root", os.path.join(T, "permroot"), ["cannot be read", "permission", "chmod"]),
    ("unreadable class folder", os.path.join(T, "permclass"), ["cannot be read", "permission", "chmod"]),
]:
    at = run_page(PAGE, lambda a, p=path: a.session_state.__setitem__(dp("dataset", "selected"), p))
    msg = blob(at)
    ok = not at.exception and any(e in msg for e in expect)
    record("perm", name, ok,
           "" if ok else f"exc={[e.value for e in at.exception][:1]} msg={msg[:140]!r}")

sys.exit(summary("UI TEST 1 · folder picker"))

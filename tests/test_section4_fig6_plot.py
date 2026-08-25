"""Tests for plot_section4_fig6.py -- the compact Figure-6 line-plot
candidate.  The plot reads ONLY the analyzer's fig6 CSV/JSON, so these
tests feed it (a) a hand-written CSV with the analyzer's columns and (b)
the real output of analyze_section4_gate.py --wave fig6 on the synthetic
192-cell grid from tests/test_section4_gate_analyzer.py.  Run with

    USE_TF=0 python -m pytest tests/test_section4_fig6_plot.py -q
"""
from __future__ import annotations

import csv
import importlib.util
import json
import os
import sys

import pytest

REPO = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
PIPE = os.path.join(REPO, "experiments", "scripts", "cluster_pipelines")
PLOT_PATH = os.path.join(PIPE, "plot_section4_fig6.py")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/perfsim-s4fig6-test-mpl")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


PLOT = _load("_plot_s4fig6_t", PLOT_PATH)

EAS = [0.0, 0.1, 0.3, 1.0]
ESS = [0.0, 0.1, 0.3, 1.0]
T_A = "t_a_evolving_minus_fixed"
G = "g_sft_minus_icl"
UNSETTLED = ("b0", 1.0, 0.3)
ABSENT = ("d8", 0.3, 1.0)


def write_csv(path, rows):
    cols = []
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def synth_analysis_dir(tmp_path):
    """A fig6 source-effect CSV with the analyzer's columns: 32 series,
    one unsettled (extend_to_60), one incomplete (NA), ea=0 identical for
    both methods, d8 served map quantized to one value."""
    d = os.path.join(str(tmp_path), "analysis")
    os.makedirs(d, exist_ok=True)
    rows = []
    for arm in ("b0", "d8"):
        for ea in EAS:
            for es in ESS:
                if ea == 0.0:
                    m = 0.004 * es                       # twin-derived
                else:
                    m = (0.02 if arm == "b0" else 0.01) * ea * (1 + es)
                half = 0.003
                key = (arm, ea, es)
                incomplete = key == ABSENT
                row = {"arm": arm, "arm_label": arm, "eps_ai": ea,
                       "eps_social": es,
                       "n_seeds_paired": 2 if incomplete else 3,
                       "status": "incomplete" if incomplete else "complete",
                       "settled": key != UNSETTLED and not incomplete,
                       "outcome": ("extend_to_60" if key == UNSETTLED
                                   else "incomplete" if incomplete
                                   else "equilibrium"),
                       "horizon": 30,
                       f"{T_A}_mean": "NA" if incomplete else m,
                       f"{T_A}_ci_lo": "NA" if incomplete else m - half,
                       f"{T_A}_ci_hi": "NA" if incomplete else m + half,
                       f"{T_A}_ci_excludes_zero": (
                           "NA" if incomplete else (abs(m) > half)),
                       "served_distinct_fixed_min": (
                           "NA" if ea == 0.0 else 1 if arm == "d8" else 7),
                       "served_distinct_evolving_min": (
                           "NA" if ea == 0.0 else 1 if arm == "d8" else 7)}
                rows.append(row)
    write_csv(os.path.join(d, PLOT.SOURCE_CSV), rows)
    # the paired method gap, G = T_a(b0) - T_a(d8), from the same rows
    gap = []
    for ea in EAS:
        for es in ESS:
            b0 = [r for r in rows if r["arm"] == "b0" and r["eps_ai"] == ea
                  and r["eps_social"] == es][0]
            d8 = [r for r in rows if r["arm"] == "d8" and r["eps_ai"] == ea
                  and r["eps_social"] == es][0]
            inc = b0["status"] != "complete" or d8["status"] != "complete"
            g = None if inc else b0[f"{T_A}_mean"] - d8[f"{T_A}_mean"]
            settled = (not inc) and b0["settled"] and d8["settled"]
            gap.append({"eps_ai": ea, "eps_social": es, "arms": "b0-d8",
                        "status": "incomplete" if inc else "complete",
                        "settled": settled,
                        "outcome": ("equilibrium" if settled else
                                    "incomplete" if inc else "extend_to_60"),
                        f"{G}_mean": "NA" if inc else g,
                        f"{G}_ci_lo": "NA" if inc else g - 0.002,
                        f"{G}_ci_hi": "NA" if inc else g + 0.002,
                        f"{G}_excludes_zero": "NA" if inc else abs(g) > 0.002})
    write_csv(os.path.join(d, PLOT.GAP_CSV), gap)
    with open(os.path.join(d, PLOT.SUMMARY_JSON), "w") as fh:
        json.dump({"primary_column": T_A,
                   "t_a_sign": "EVOLVING MINUS FIXED (synthetic)",
                   "g_sign": "positive G = SFT exceeds ICL (synthetic)",
                   "gate_info": "ok (synthetic)",
                   "coverage_note": "192/192 cells present (synthetic)"},
                  fh)
    return d


@pytest.fixture(scope="module")
def synth(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fig6plot")
    adir = synth_analysis_dir(tmp)
    out = os.path.join(str(tmp), "previews")
    return adir, out


@pytest.fixture(scope="module")
def plotted(synth):
    adir, out = synth
    rc = PLOT.main(["--analysis-dir", adir, "--out-dir", out])
    return rc, out


def test_figure_files_exist(plotted):
    rc, out = plotted
    assert rc == 0
    for name in ("section4_fig6_candidate.pdf",
                 "section4_fig6_candidate.png",
                 "section4_fig6_candidate_2panel.pdf",
                 "section4_fig6_candidate_2panel.png",
                 "section4_fig6_candidate_gap.pdf",
                 "section4_fig6_candidate_gap.png",
                 "section4_fig6_candidate_caption.txt"):
        p = os.path.join(out, name)
        assert os.path.exists(p) and os.path.getsize(p) > 0, name


def test_no_title_text_and_no_trajectory_access_in_source():
    src = open(PLOT_PATH).read()
    assert "set_title(" not in src
    assert "suptitle(" not in src
    assert "no set_title" in src              # the convention is stated
    assert "import torch" not in src
    assert "trajectory.pt" not in src
    assert "torch.load" not in src
    assert 'matplotlib.use("Agg")' in src


def test_refuses_an_out_dir_under_paper(synth, tmp_path):
    adir, _ = synth
    bad = os.path.join(str(tmp_path), "paper", "figures")
    with pytest.raises(SystemExit) as e:
        PLOT.main(["--analysis-dir", adir, "--out-dir", bad])
    assert e.value.code == 1
    assert not os.path.exists(bad)
    assert "paper" in PLOT.DEFAULT_OUT.split(os.sep) is False or \
        "paper" not in PLOT.DEFAULT_OUT.split(os.sep)
    assert PLOT.DEFAULT_OUT.endswith(os.path.join("notes", "pofd",
                                                  "section4_fig6",
                                                  "previews"))


def test_unsettled_cell_is_ringed_and_named_in_the_caption(plotted):
    _, out = plotted
    text = open(os.path.join(out, "section4_fig6_candidate_caption.txt")
                ).read()
    assert "RINGED = UNSETTLED (1;" in text
    assert "SFT eps_AI=1 eps_social=.3 [extend_to_60]" in text
    assert "NOT DRAWN (1" in text
    assert "personal-history ICL eps_AI=.3 eps_social=1 [incomplete]" in text
    assert "EVOLVING MINUS" in text
    assert "IDENTICAL for both methods" in text
    assert "SERVED-VALUE QUANTIZATION" in text
    assert "no title text" in text
    assert "Gate verdict: ok (synthetic)" in text
    # the gap block: G at eps_AI = 0 is the anchor, the unsettled gap
    # point is ringed and named, the incomplete one is not drawn
    assert "PAIRED METHOD GAP" in text
    assert "G is IDENTICALLY 0" in text
    assert "RINGED = UNSETTLED gap points (1): eps_AI=1 eps_social=.3 " \
        "[extend_to_60]" in text
    assert "Gap points NOT DRAWN (1): eps_AI=.3 eps_social=1 [incomplete]" \
        in text


def test_categorical_labels_and_ramp():
    assert [PLOT.cat_label(e) for e in ESS] == ["0", ".1", ".3", "1"]
    assert len(PLOT.EA_RAMP) == 4
    # one hue, light -> dark: luminance strictly decreasing
    def lum(h):
        r, g, b = (int(h[i:i + 2], 16) for i in (1, 3, 5))
        return 0.2126 * r + 0.7152 * g + 0.0722 * b
    lums = [lum(c) for c in PLOT.EA_RAMP]
    assert all(a > b for a, b in zip(lums, lums[1:]))


def test_reader_drops_absent_and_flags_unsettled_without_inventing(synth):
    adir, _ = synth
    rows = PLOT.read_rows(adir)
    assert len(rows) == 32
    drawable, absent = PLOT.classify(rows)
    assert len(drawable) == 31 and len(absent) == 1
    assert (absent[0]["arm"], absent[0]["eps_ai"],
            absent[0]["eps_social"]) == ABSENT
    assert absent[0]["mean"] is None                 # never synthesised
    uns = [r for r in drawable if not r["settled"]]
    assert [(r["arm"], r["eps_ai"], r["eps_social"]) for r in uns] == \
        [UNSETTLED]
    # ea=0 drawable for BOTH methods, identical
    for es in ESS:
        b0 = [r for r in drawable if r["arm"] == "b0" and r["eps_ai"] == 0
              and r["eps_social"] == es][0]
        d8 = [r for r in drawable if r["arm"] == "d8" and r["eps_ai"] == 0
              and r["eps_social"] == es][0]
        assert b0["mean"] == d8["mean"]


def test_draw_panel_rings_only_the_unsettled_point(synth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    adir, _ = synth
    rows = PLOT.read_rows(adir)
    drawable, _ = PLOT.classify(rows)
    colors = {ea: PLOT.EA_RAMP[i] for i, ea in enumerate(EAS)}
    fig, ax = plt.subplots()
    ringed = PLOT.draw_panel(ax, drawable, ESS, EAS, ["b0", "d8"], colors)
    plt.close(fig)
    assert [(a, ea, es) for a, ea, es, _ in ringed] == [UNSETTLED]
    assert ringed[0][3] == "extend_to_60"


def test_missing_csv_is_a_clean_failure(tmp_path):
    with pytest.raises(SystemExit) as e:
        PLOT.main(["--analysis-dir", str(tmp_path), "--out-dir",
                   str(tmp_path / "o")])
    assert e.value.code == 1


def test_end_to_end_from_the_real_analyzer_output(tmp_path, capsys):
    """analyze_section4_gate.py --wave fig6 on the synthetic 192-cell
    grid, then the plot on its CSV/JSON: the unsettled pair the analyzer
    found is the one the caption names."""
    T6 = _load("_t_s4gate_for_plot",
               os.path.join(REPO, "tests", "test_section4_gate_analyzer.py"))
    root = os.path.join(str(tmp_path), "runs")
    T6.build_fig6_grid(root)
    gate = T6._gate_json(os.path.join(str(tmp_path), "gate.json"))
    adir = os.path.join(str(tmp_path), "analysis")
    rc = T6.AN.main(["--wave", "fig6", "--run-root", root, "--out-dir",
                     adir, "--gate-json", gate, "--no-figs"])
    assert rc == 2
    out = os.path.join(str(tmp_path), "previews")
    assert PLOT.main(["--analysis-dir", adir, "--out-dir", out]) == 0
    printed = " ".join(capsys.readouterr().out.split())   # unwrap
    text = open(os.path.join(out, "section4_fig6_candidate_caption.txt")
                ).read()
    assert "RINGED = UNSETTLED (1;" in printed
    arm, ea, es, _ = T6.DRIFT_PAIR
    name = (f"{PLOT.ARM_LABEL[arm]} eps_AI={PLOT.cat_label(ea)} "
            f"eps_social={PLOT.cat_label(es)} [extend_to_60]")
    assert name in printed and name in text
    assert "No point is ringed" not in printed
    assert "NOT DRAWN" not in printed
    assert "SERVED-VALUE QUANTIZATION" in printed
    assert "EVOLVING MINUS FIXED" in printed
    for name in ("section4_fig6_candidate.pdf",
                 "section4_fig6_candidate_2panel.png"):
        assert os.path.getsize(os.path.join(out, name)) > 0


def test_gap_reader_and_zero_anchor(synth):
    adir, _ = synth
    gap = PLOT.read_gap_rows(adir)
    assert len(gap) == 16
    drawable, absent = PLOT.classify(gap)
    assert len(drawable) == 15 and len(absent) == 1
    for r in drawable:
        if r["eps_ai"] == 0.0:
            assert r["mean"] == 0.0 and r["settled"]
    assert [(r["eps_ai"], r["eps_social"]) for r in drawable
            if not r["settled"]] == [UNSETTLED[1:]]


def test_missing_gap_csv_is_a_clean_failure(synth, tmp_path):
    adir, _ = synth
    import shutil
    d2 = os.path.join(str(tmp_path), "nogap")
    shutil.copytree(adir, d2)
    os.remove(os.path.join(d2, PLOT.GAP_CSV))
    with pytest.raises(SystemExit) as e:
        PLOT.main(["--analysis-dir", d2, "--out-dir",
                   os.path.join(str(tmp_path), "o")])
    assert e.value.code == 1

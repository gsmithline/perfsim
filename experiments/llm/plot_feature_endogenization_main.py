#!/usr/bin/env python3
"""Main-paper feature-endogenization figure.

Panel (a) compares retention strengths and the frozen model. Panel (b)
isolates the feature channel with removal and permutation controls. Panel (c)
uses the controlled reference intervention to identify the source of the
feature dependence. All panels use the matched peer environment.

Panels (a) and (b) run on FIVE seeds (2026-08-19): {0, 42, 43, 44, 45},
plotted as the five-seed mean with a 95% Student-t ribbon (df = 4). All
five are HARD-REQUIRED -- a missing run is an error naming the tag, never
a silently smaller average. Panel (c) is unchanged: the controlled
reference intervention already ran at five seeds and keeps its own loader.

The runner spells the retention coefficient kl_beta and the run tags spell
it b<...>; every DISPLAYED label here -- legend entries and the condition
column of the exported CSVs -- calls it lambda, the paper's notation.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_feature_endogenization_beta_final import (
    RUN_ROOT,
    TASTE_COLUMNS,
    control_tag,
    incremental_r2,
    load_run,
    natural_tag,
)
from plot_teacher_intervention_appendix import (
    mean_ci,
    load_runs as load_teacher_runs,
    stack_series as stack_teacher_series,
    teacher_constant,
)


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "paper" / "figures"
ROUNDS = np.arange(30)
# panels (a)/(b): the three established seeds plus the 2026-08-19
# extension. Deliberately NOT the shared SEEDS constant, which the
# beta-sweep and environment-dose figures still use at three seeds.
PANEL_SEEDS = (0, 42, 43, 44, 45)


def require_runs(seeds: tuple[int, ...]) -> None:
    """Fail loudly, naming every missing tag, before any plotting.

    A silently smaller seed set would change the mean and the interval
    without changing the figure's appearance, so absence is an error.
    """
    missing = []
    for seed in seeds:
        tags = [natural_tag(beta, seed) for beta in (0.0, 0.5, 1.0)]
        tags += [
            control_tag(name, seed)
            for name in ("frozen", "gender_removed", "gender_randomized")
        ]
        for tag in tags:
            if not (RUN_ROOT / tag / "trajectory.pt").exists():
                missing.append(tag)
    if missing:
        raise SystemExit(
            f"feature_endogenization_main: {len(missing)} of "
            f"{6 * len(seeds)} required runs are missing for seeds "
            f"{list(seeds)} -- panels (a)/(b) hard-require all of them:\n  "
            + "\n  ".join(missing)
        )


def series_for_run(
    run: dict,
    tastes: np.ndarray,
    feature: np.ndarray,
    source: str = "population",
) -> np.ndarray:
    key = "op_raw" if source == "population" else "twin_raw"
    return np.asarray(
        [
            incremental_r2(tastes, feature, np.asarray(opinion, dtype=float))
            for opinion in run["trajectory"][key]
        ],
        dtype=float,
    )


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    require_runs(PANEL_SEEDS)

    reference = load_run(natural_tag(1.0, 0))["trajectory"]
    tastes = np.column_stack(
        [np.asarray(reference["profiles"][key], dtype=float) for key in TASTE_COLUMNS]
    )
    true_gender = np.asarray(reference["profiles"]["gender"]) == "M"

    panel_a: dict[str, dict[int, np.ndarray]] = {
        "lambda_1": {},
        "frozen": {},
        "lambda_0.5": {},
        "lambda_0": {},
        "no_platform": {},
    }
    panel_b: dict[str, dict[int, np.ndarray]] = {
        "natural": {},
        "removed": {},
        "permuted_true": {},
        "permuted_displayed": {},
    }
    for seed in PANEL_SEEDS:
        natural_runs = {
            beta: load_run(natural_tag(beta, seed))
            for beta in (0.0, 0.5, 1.0)
        }
        panel_a["lambda_1"][seed] = series_for_run(
            natural_runs[1.0], tastes, true_gender
        )
        panel_a["lambda_0.5"][seed] = series_for_run(
            natural_runs[0.5], tastes, true_gender
        )
        panel_a["lambda_0"][seed] = series_for_run(
            natural_runs[0.0], tastes, true_gender
        )
        panel_a["no_platform"][seed] = series_for_run(
            natural_runs[0.0], tastes, true_gender, source="twin"
        )

        frozen = load_run(control_tag("frozen", seed))
        panel_a["frozen"][seed] = series_for_run(frozen, tastes, true_gender)

        removed = load_run(control_tag("gender_removed", seed))
        permuted = load_run(control_tag("gender_randomized", seed))
        displayed_gender = np.asarray(
            permuted["trajectory"]["profiles"]["gender"]
        ) == "M"

        panel_b["natural"][seed] = panel_a["lambda_1"][seed]
        panel_b["removed"][seed] = series_for_run(removed, tastes, true_gender)
        panel_b["permuted_true"][seed] = series_for_run(
            permuted, tastes, true_gender
        )
        panel_b["permuted_displayed"][seed] = series_for_run(
            permuted, tastes, displayed_gender
        )
    teacher_runs = load_teacher_runs()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8.5,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.15, 2.55), layout="constrained")
    grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.06))
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[0, 1]),
        fig.add_subplot(grid[0, 2]),
    )

    styles_a = (
        ("lambda_1", r"$\lambda=1$", "#0072B2", "-", 1.55),
        ("lambda_0.5", r"$\lambda=.5$", "#CC79A7", "--", 1.1),
        ("lambda_0", r"$\lambda=0$", "#009E73", "-", 1.1),
        ("frozen", "frozen", "#D55E00", "-", 1.45),
        ("no_platform", "no platform", "#777777", ":", 1.1),
    )
    styles_b = (
        ("natural", "natural", "#0072B2", "-", 1.55),
        ("removed", "removed", "#009E73", "--", 1.15),
        ("permuted_true", "permuted: true", "#777777", ":", 1.15),
        (
            "permuted_displayed",
            "permuted: shown",
            "#CC79A7",
            "-.",
            1.3,
        ),
    )
    rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for axis, title, data, styles, panel, panel_seeds in (
        (
            axes[0],
            "(a) Association over time",
            panel_a,
            styles_a,
            "a",
            PANEL_SEEDS,
        ),
        (axes[1], "(b) Feature controls", panel_b, styles_b, "b",
         PANEL_SEEDS),
    ):
        axis.axhline(0, color="#999999", linewidth=0.7, zorder=0)
        for key, label, color, linestyle, linewidth in styles:
            values = np.stack([data[key][seed] for seed in panel_seeds])
            # five-seed mean with the 95% Student-t interval (df = 4)
            mean, interval = mean_ci(values)
            axis.fill_between(
                ROUNDS,
                mean - interval,
                mean + interval,
                color=color,
                alpha=0.16,
                linewidth=0,
                zorder=1,
            )
            axis.plot(
                ROUNDS,
                mean,
                color=color,
                linestyle=linestyle,
                linewidth=linewidth,
                label=label,
                zorder=2,
            )
            for seed_index, seed in enumerate(panel_seeds):
                for round_index, value in enumerate(values[seed_index]):
                    rows.append(
                        {
                            "panel": panel,
                            "condition": key,
                            "seed": seed,
                            "round": round_index,
                            "population_incremental_r2": value,
                        }
                    )
            for round_index in range(values.shape[1]):
                summary_rows.append(
                    {
                        "panel": panel,
                        "condition": key,
                        "round": round_index,
                        "n_seeds": values.shape[0],
                        "mean": mean[round_index],
                        "ci_half_width": interval[round_index],
                        "ci_lo": mean[round_index] - interval[round_index],
                        "ci_hi": mean[round_index] + interval[round_index],
                    }
                )
        axis.set_title(title, pad=29)
        axis.set_xlabel("round")
        axis.set_xlim(0, 29)
        axis.set_xticks([0, 10, 20, 29])
        axis.set_ylim(-0.003, 0.037)
        axis.set_yticks([0, 0.01, 0.02, 0.03])
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.01),
            fontsize=5.9,
            ncol=3 if panel == "a" else 2,
            columnspacing=0.65,
            handlelength=1.8,
            handletextpad=0.35,
            borderaxespad=0,
        )

    axes[0].set_ylabel(r"Incremental $R^2$ of gender")
    axes[1].tick_params(labelleft=False)

    source_axis = axes[2]
    source_axis.axhline(0, color="#999999", linewidth=0.75, zorder=0)
    stages = np.arange(3)
    for arm, label, color, marker in (
        ("tpos", "positive", "#0072B2", "o"),
        ("tneu", "neutral", "#777777", "s"),
        ("tneg", "negative", "#D55E00", "^"),
    ):
        teacher = stack_teacher_series(teacher_runs, arm, "gg_teacher")
        student = stack_teacher_series(teacher_runs, arm, "gg_pred_true")
        population = stack_teacher_series(teacher_runs, arm, "gg_op_true")
        student_mean, student_ci = mean_ci(student)
        population_mean, population_ci = mean_ci(population)
        means = np.asarray(
            [teacher_constant(teacher), student_mean[-1], population_mean[-1]]
        )
        intervals = np.asarray([0, student_ci[-1], population_ci[-1]])

        source_axis.plot(
            stages,
            means,
            color=color,
            linewidth=1.25,
            alpha=0.85,
            zorder=2,
        )
        source_axis.errorbar(
            stages,
            means,
            yerr=intervals,
            fmt=marker,
            color=color,
            markerfacecolor="white" if arm == "tneu" else color,
            markersize=4.6,
            linewidth=1.0,
            capsize=2.3,
            label=label,
            zorder=3,
        )

    source_axis.set_title("(c) Reference source check", pad=29)
    source_axis.set_xticks(stages, ["reference", "student", "population"])
    source_axis.set_ylabel("Gender gap (M - F)")
    source_axis.set_ylim(-0.18, 0.19)
    source_axis.set_yticks([-0.15, 0, 0.15])
    source_axis.spines[["top", "right"]].set_visible(False)
    source_axis.legend(
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        fontsize=5.9,
        ncol=3,
        columnspacing=0.75,
        handletextpad=0.25,
        borderaxespad=0,
    )

    for extension in ("pdf", "png"):
        destination = OUT_ROOT / f"feature_endogenization_main.{extension}"
        fig.savefig(destination, dpi=300 if extension == "png" else None)
    plt.close(fig)

    csv_path = OUT_ROOT / "feature_endogenization_main_points.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary_path = OUT_ROOT / "feature_endogenization_main_summary.csv"
    with summary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    # Report the FULL five-seed mean trajectory, then locate the peak on
    # that curve. The peak is never chosen first and never re-selected
    # against a different seed set or window.
    print(f"seeds = {list(PANEL_SEEDS)} (n={len(PANEL_SEEDS)}), "
          "95% Student-t intervals, df=4")
    for key in (
        "lambda_1",
        "frozen",
        "lambda_0.5",
        "lambda_0",
        "no_platform",
    ):
        values = np.stack([panel_a[key][seed] for seed in PANEL_SEEDS])
        mean, interval = mean_ci(values)
        peak = int(mean.argmax())
        print(f"\n{key}: full five-seed mean trajectory")
        print(
            "  "
            + "  ".join(
                f"r{r:02d}={mean[r]:+.5f}" for r in range(len(mean))
            )
        )
        print(
            f"  peak={mean[peak]:+.6f} +/- {interval[peak]:.6f} "
            f"at round {peak}; final={mean[-1]:+.6f} "
            f"+/- {interval[-1]:.6f}"
        )
    for key in ("natural", "removed", "permuted_true", "permuted_displayed"):
        values = np.stack([panel_b[key][seed] for seed in PANEL_SEEDS])
        mean, interval = mean_ci(values)
        print(
            f"{key}: rounds8-13={values[:, 8:14].mean():+.6f}, "
            f"final={mean[-1]:+.6f} +/- {interval[-1]:.6f}"
        )


if __name__ == "__main__":
    main()

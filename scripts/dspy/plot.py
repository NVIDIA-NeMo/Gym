import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description="Plot a GEPA optimization curve.")
    p.add_argument("results", help="gepa_results.json from optimize.py")
    p.add_argument("-o", "--out", default="gepa_curve.png")
    args = p.parse_args()

    d = json.load(open(args.results))
    curve = d.get("curve") or []
    if not curve:
        raise SystemExit("No curve in results (MIPRO has none, only GEPA exports a curve in the recipe).")

    use_calls = all(c.get("eval_calls") is not None for c in curve)
    xs = [c["eval_calls"] if use_calls else c["candidate"] for c in curve]
    topline = [c["topline"] for c in curve]
    points = [c["val_acc"] for c in curve]

    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.3})
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(xs, points, s=22, color="#94a3b8", zorder=2, label="candidate")
    ax.plot(xs, topline, drawstyle="steps-post", lw=2.2, color="#2563eb", zorder=3, label="best so far")
    ax.axhline(d["baseline_val_acc"], ls="--", lw=1.2, color="#dc2626", label="seed baseline")

    ax.set_xlabel("optimization step" + (" (rollouts)" if use_calls else ""))
    ax.set_ylabel("validation accuracy")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title(f"GEPA prompt optimization: {d['baseline_val_acc']:.2f} -> {d['optimized_val_acc']:.2f}")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

"""
Aggregate results/methods_grid.csv into the per-method comparison.

For each training-free method vs base, across all (encoder x dataset) points:
  - mean metric + mean delta vs base (acc/f1: higher better; ece/aurc: lower better)
  - never-harm: # points where acc not below base (the deploy guarantee)
  - wins: # points strictly better on the metric
Broken down by task family (multiclass defect/poj vs binary vuln) and kind.

Pure stdlib + numpy. Prints tables; writes results/methods_summary.json.
"""
import os, csv, json, collections
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV = os.path.join(REPO, "results", "methods_grid.csv")
METHODS = ["base", "temp", "logit_adj", "knn", "gate", "temp_gate"]
LOWER_BETTER = {"ece", "aurc"}
BINARY_DS = {"devign", "primevul", "reveal", "diversevul", "vuln"}  # binary vuln ("vuln"=fine-tuned Devign anchor)
METRICS = ["acc", "f1", "ece", "aurc"]


def load():
    rows = collections.defaultdict(dict)   # point -> method -> {metrics}
    meta = {}
    with open(CSV) as f:
        for r in csv.DictReader(f):
            p = r["point"]
            rows[p][r["method"]] = {m: float(r[m]) for m in METRICS}
            meta[p] = {"dataset": r["dataset"], "kind": r["kind"],
                       "C": int(r["C"]), "imbalance": float(r["imbalance"])}
    return rows, meta


def family(ds):
    return "binary_vuln" if ds in BINARY_DS else "multiclass"


def summarize(points, rows, meta):
    """points: list of point names. Returns per-method aggregates."""
    out = {}
    for meth in METHODS:
        agg = {}
        for m in METRICS:
            vals, deltas = [], []
            for p in points:
                if meth not in rows[p] or "base" not in rows[p]:
                    continue
                v = rows[p][meth][m]; b = rows[p]["base"][m]
                vals.append(v)
                d = (b - v) if m in LOWER_BETTER else (v - b)   # positive = better
                deltas.append(d)
            agg[m] = {"mean": round(float(np.mean(vals)), 4),
                      "mean_delta": round(float(np.mean(deltas)), 4),
                      "wins": int(sum(d > 1e-9 for d in deltas)),
                      "n": len(deltas)}
        # never-harm: acc delta >= -0.002 (tolerance for ties)
        acc_d = [rows[p][meth]["acc"] - rows[p]["base"]["acc"] for p in points
                 if meth in rows[p]]
        agg["never_harm_acc"] = int(sum(d >= -0.002 for d in acc_d))
        agg["n_points"] = len(acc_d)
        out[meth] = agg
    return out


def fmt_block(title, S):
    print(f"\n{'='*78}\n{title}\n{'='*78}")
    print(f"{'method':<11}" + "".join(f"{m+' Δ':>14}" for m in METRICS) + f"{'neverharm':>12}")
    for meth in METHODS:
        a = S[meth]
        line = f"{meth:<11}"
        for m in METRICS:
            d = a[m]["mean_delta"]; w = a[m]["wins"]; n = a[m]["n"]
            line += f"{d:+.4f}({w}/{n})".rjust(14)
        line += f"{a['never_harm_acc']}/{a['n_points']}".rjust(12)
        print(line)
    print("Δ = mean improvement vs base (+ is better; ece/aurc sign-flipped). (wins/total).")


def main():
    rows, meta = load()
    allp = sorted(rows)
    print(f"{len(allp)} points loaded")
    full = {"all": summarize(allp, rows, meta)}
    fmt_block(f"ALL POINTS (n={len(allp)})", full["all"])

    for fam in ["multiclass", "binary_vuln"]:
        pts = [p for p in allp if family(meta[p]["dataset"]) == fam]
        if pts:
            full[fam] = summarize(pts, rows, meta)
            fmt_block(f"{fam}  (n={len(pts)}: {sorted({meta[p]['dataset'] for p in pts})})", full[fam])

    for kind in ["fine_tuned", "frozen_probe"]:
        pts = [p for p in allp if meta[p]["kind"] == kind]
        if pts:
            full[kind] = summarize(pts, rows, meta)
            fmt_block(f"{kind}  (n={len(pts)})", full[kind])

    # headline numbers
    S = full["all"]
    print(f"\n{'='*78}\nHEADLINES\n{'='*78}")
    base_ece = np.mean([rows[p]["base"]["ece"] for p in allp])
    temp_ece = np.mean([rows[p]["temp"]["ece"] for p in allp])
    print(f"  temp ECE: {base_ece:.4f} -> {temp_ece:.4f}  ({base_ece/max(temp_ece,1e-9):.1f}x reduction), "
          f"wins {S['temp']['ece']['wins']}/{S['temp']['ece']['n']}")
    print(f"  gate acc Δ: {S['gate']['acc']['mean_delta']:+.4f}, never-harm {S['gate']['never_harm_acc']}/{S['gate']['n_points']}, "
          f"AURC Δ {S['gate']['aurc']['mean_delta']:+.4f}")
    if "multiclass" in full:
        print(f"  gate acc Δ on multiclass: {full['multiclass']['gate']['acc']['mean_delta']:+.4f} "
              f"(wins {full['multiclass']['gate']['acc']['wins']}/{full['multiclass']['gate']['acc']['n']})")
    print(f"  logit_adj F1 Δ: {S['logit_adj']['f1']['mean_delta']:+.4f}")

    json.dump(full, open(os.path.join(REPO, "results", "methods_summary.json"), "w"), indent=2)
    print("\nsaved -> results/methods_summary.json")


if __name__ == "__main__":
    main()

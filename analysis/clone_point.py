"""
Run the method grid (base/temp/logit_adj/knn/gate) on the clone caches and report
whether retrieval/gating helps on clone detection.

Clone detection is binary but separable. If gate/knn beat base here, the determining
factor is separability rather than task arity (binary vulnerability fails because it is
unlearnable, not because it is binary). Appends rows to results/methods_grid.csv.

Run after embed_clone.py.  python analysis/clone_point.py
"""
import os, sys
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kNN-Prediction"))
import numpy as np
from methods_grid import frozen_point, eval_methods, append_rows, imbalance_ratio, done
import run_grid as G

DS = "bigclone"


def main():
    have = done()
    summary = []
    for enc in G.ENCODERS:
        point = f"{enc}/{DS}"
        cache = os.path.join(G.EMB_DIR, f"{enc}__{DS}.npz")
        if not os.path.exists(cache):
            print(f"[MISS] {cache} — run embed_clone.py first"); continue
        P = frozen_point(enc, DS, {})
        if P is None:
            print(f"[SKIP {point}] cache incomplete"); continue
        rows = eval_methods(P["va_lg"], P["va_l"], P["te_lg"], P["te_l"],
                            P["knn_va"], P["knn_te"], P["prior"], P["C"])
        if point not in have:
            append_rows(point, enc, DS, P["kind"], P["C"], P["n_test"],
                        imbalance_ratio(P["te_l"]), rows)
        b = {r[0]: r[1] for r in rows}
        summary.append((point, b))
        print(f"[{point}] base acc={b['base']['acc']} | knn acc={b['knn']['acc']} "
              f"| gate acc={b['gate']['acc']} | base aurc={b['base']['aurc']} gate aurc={b['gate']['aurc']}")

    if summary:
        d_knn = np.mean([b["knn"]["acc"] - b["base"]["acc"] for _, b in summary])
        d_gate = np.mean([b["gate"]["acc"] - b["base"]["acc"] for _, b in summary])
        d_aurc = np.mean([b["base"]["aurc"] - b["gate"]["aurc"] for _, b in summary])
        print(f"\n=== CLONE (binary, separable) n={len(summary)} ===")
        print(f"  base acc mean: {np.mean([b['base']['acc'] for _,b in summary]):.4f}")
        print(f"  kNN  acc delta vs base: {d_knn:+.4f}")
        print(f"  gate acc delta vs base: {d_gate:+.4f}")
        print(f"  gate AURC improvement:  {d_aurc:+.4f}")
        verdict = "HELPS -> axis = SEPARABILITY (binary can win)" if (d_knn > 0.003 or d_gate > 0.003) \
            else "no help -> axis stays arity OR clone too easy (ceiling)"
        print(f"  VERDICT: {verdict}")


if __name__ == "__main__":
    main()

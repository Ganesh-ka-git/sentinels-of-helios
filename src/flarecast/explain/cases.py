"""Case selection for the Grad-CAM figures (decisions.md §11, Change Log 13).

Cases come from the TEST predictions of ONE reference model (E8, seed 0) at its validation-tuned tau1 and are then
reused for every model. Categories (distinct HARPs, deterministic rules, no manual picking):
  TP-X  true positive whose true class is X: highest P(>=M) among the X true positives
  TP-M  true positive with true class M: highest P(>=M) among the M true positives (other HARP)
  FN    two missed events (true >=M, P(>=M) < tau1): lowest P(>=M) first
  FA    two false alarms (true <M, P(>=M) >= tau1): highest P(>=M) first
  TN    one true negative (true <M, P(>=M) < tau1): lowest P(>=M)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CASE_SPEC = [("TP-X", 1), ("TP-M", 1), ("FN", 2), ("FA", 2), ("TN", 1)]   # (category, count) = 7 cases
CLASS_NAMES = {0: "<M", 1: "M", 2: "X"}
CASE_COLUMNS = ["case", "harpnum", "t", "true_class", "e8_p_ge_m", "category", "e8_tau1"]


def select_cases(preds: pd.DataFrame, tau1: float) -> pd.DataFrame:
    """``preds`` = test predictions (columns harpnum, t, label, p1, p2) of the reference model; returns 7 cases."""
    d = preds.copy()
    d["s"] = d["p1"] + d["p2"]
    pos, hit = d["label"] >= 1, d["s"] >= tau1
    pools = {
        "TP-X": d[pos & hit & (d["label"] == 2)].sort_values(["s", "harpnum", "t"], ascending=[False, True, True]),
        "TP-M": d[pos & hit & (d["label"] == 1)].sort_values(["s", "harpnum", "t"], ascending=[False, True, True]),
        "FN": d[pos & ~hit].sort_values(["s", "harpnum", "t"], ascending=[True, True, True]),
        "FA": d[~pos & hit].sort_values(["s", "harpnum", "t"], ascending=[False, True, True]),
        "TN": d[~pos & ~hit].sort_values(["s", "harpnum", "t"], ascending=[True, True, True]),
    }
    rows, used = [], set()
    for cat, n in CASE_SPEC:
        pool = pools[cat]
        got = 0
        for r in pool.itertuples():
            if int(r.harpnum) in used:
                continue
            used.add(int(r.harpnum))
            rows.append({"case": f"{len(rows) + 1:02d}_{cat}", "harpnum": int(r.harpnum),
                         "t": pd.Timestamp(r.t).strftime("%Y-%m-%d %H:%M"), "true_class": CLASS_NAMES[int(r.label)],
                         "e8_p_ge_m": float(r.s), "category": cat, "e8_tau1": float(tau1)})
            got += 1
            if got == n:
                break
        if got < n:
            raise ValueError(f"Only {got} of {n} requested {cat} cases available with distinct HARPs "
                             f"(pool size {len(pool)}, tau1={tau1})")
    out = pd.DataFrame(rows, columns=CASE_COLUMNS)
    if len(out) != sum(n for _, n in CASE_SPEC):
        raise AssertionError("wrong number of cases")
    return out

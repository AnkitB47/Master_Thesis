import json
import os
import sys

import pandas as pd


def verify_kpis(
    path: str = "results_1d_nominal/pfr_kpis.csv",
    pv: float = 0.15,
    delay: float = 0.15,
    resid: float = 0.05,
    timescale: float = 0.15,
    conv: float = 0.10,
    h2co: float = 0.15,
) -> None:
    df = pd.read_csv(path)
    if df.empty:
        print("No KPI data found.")
        sys.exit(1)
    fails: list[str] = []
    for _, row in df.iterrows():
        if (
            abs(row.get("pv_err", 0)) > pv
            or abs(row.get("delay_metric", 0)) > delay
            or abs(row.get("pen_species", 0)) > resid
            or abs(row.get("tau_mis", 0)) > timescale
            or abs(row.get("CH4_full", 0) - row.get("CH4_red", 0)) > conv
            or abs(row.get("H2CO_full", 0) - row.get("H2CO_red", 0)) > h2co
        ):
            fails.append(row.get("case_id", "?"))
    if fails:
        print("❌ KPI violations:", ", ".join(fails))
        sys.exit(2)
    print("✅ All KPIs within tolerance")


if __name__ == "__main__":
    kwargs = json.loads(os.environ.get("KPI_TOLS", "{}"))
    verify_kpis(**kwargs)

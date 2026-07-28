"""Récupération vers la vraie épaisseur : le scan bougé, après chaque condition, se rapproche-t-il
de l'épaisseur régionale du scan immobile du même sujet ? Répond à « les mesures corticales suivent-elles ».

Épaisseur régionale (68 régions FreeSurfer), en mm. Pour chaque sujet, run bougé (run-02, run-03) et
condition, trois distances (erreur absolue moyenne sur les régions, mm) :
- mae_within  : |bougé(condition) − immobile(condition)|  -> mouvement restant, sans le biais d'offset
                (immobile et bougé subissent le même offset dans une condition).
- mae_truth   : |bougé(condition) − immobile(BRUT)|        -> erreur totale à la vraie valeur.
- offset      : |immobile(condition) − immobile(BRUT)|      -> distorsion appliquée à un scan propre.
Plus l'étendue inter-régions (écart-type sur les régions) pour repérer un aplatissement (lissage).

Vérité = épaisseur régionale du scan immobile BRUT (run-01), la moins traitée.
Sortie : results/ds004332/phase4_compare_3bras/recovery_metrics.csv
"""
import numpy as np
import pandas as pd
from pathlib import Path

HOME = Path.home()
REPO = Path(__file__).resolve().parents[3]
DERIV = HOME / "Documents/derivatives/ds004332"
OUT = REPO / "results/ds004332/phase4_compare_3bras/recovery_metrics.csv"
CONDS = ["brut", "preproc", "jdac", "jdac_antiartonly", "jdac_nodenoise"]
WIDE = {"preproc": "thickness/thickness_preproc_{h}.csv",
        "jdac": "thickness/thickness_jdac_{h}.csv",
        "jdac_antiartonly": "thickness/thickness_jdac_antiartonly_{h}.csv",
        "jdac_nodenoise": "thickness/thickness_jdac_nodenoise_{h}.csv"}


def load_regional():
    frames = []
    b = pd.read_csv(REPO / "results/ds004332/phase1_RAW/ThickAvg_phase1_complete.csv")
    b = b[b["ThickAvg"] > 0].copy()
    b["run"] = b["subject"].str.split("_").str[1]
    b["sub"] = b["subject"].str.split("_").str[0]
    b["region"] = b["hemi"] + "_" + b["region"].astype(str)
    frames.append(b[["sub", "run", "region", "ThickAvg"]].rename(columns={"ThickAvg": "th"}).assign(condition="brut"))
    for c in CONDS:
        if c == "brut":
            continue
        for h in ["lh", "rh"]:
            w = pd.read_csv(DERIV / WIDE[c].format(h=h), sep="\t")
            w = w.rename(columns={w.columns[0]: "id"})
            cols = [x for x in w.columns if x.endswith("_thickness") and "MeanThickness" not in x]
            lg = w.melt(id_vars="id", value_vars=cols, var_name="region", value_name="th")
            lg["region"] = lg["region"].str.replace("_thickness", "", regex=False)
            lg["sub"] = lg["id"].str.split("_").str[0]
            lg["run"] = lg["id"].str.split("_").str[1]
            frames.append(lg[["sub", "run", "region", "th"]].assign(condition=c))
    return pd.concat(frames, ignore_index=True)


def mae(a, b):
    common = a.index.intersection(b.index)
    return float((a[common] - b[common]).abs().mean()) if len(common) else np.nan


def main():
    df = load_regional()
    ser = {k: v.set_index("region")["th"] for k, v in df.groupby(["sub", "condition", "run"])}
    subjects = sorted(df["sub"].unique())
    rows = []
    for s in subjects:
        truth = ser.get((s, "brut", "run-01"))
        if truth is None:
            continue
        for c in CONDS:
            still_c = ser.get((s, c, "run-01"))
            offset = mae(still_c, truth) if still_c is not None else np.nan
            for run in ["run-02", "run-03"]:
                moved = ser.get((s, c, run))
                if moved is None:
                    continue
                rows.append({
                    "sub": s, "condition": c, "run": run,
                    "mae_within": mae(moved, still_c) if still_c is not None else np.nan,
                    "mae_truth": mae(moved, truth),
                    "offset": offset,
                    "sd_moved": float(moved.std()),
                    "sd_truth": float(truth.std()),
                })
    res = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT, index=False)

    print("== Moyenne sur sujets (mm), par condition x run bougé ==")
    for run in ["run-02", "run-03"]:
        print(f"\n-- {run} ({'nodding' if run=='run-02' else 'shaking'}) --")
        t = (res[res["run"] == run].groupby("condition")[["mae_within", "mae_truth", "offset", "sd_moved"]]
             .mean().reindex(CONDS))
        t["sd_truth"] = res.groupby("condition")["sd_truth"].mean().reindex(CONDS)
        print(t.round(3).to_string())

    print("\n== sub-19 en détail (mm) ==")
    d = res[res["sub"] == "sub-19"].set_index(["condition", "run"])[["mae_within", "mae_truth", "offset"]]
    print(d.round(3).to_string())
    print(f"\n{len(res)} lignes -> {OUT}")


if __name__ == "__main__":
    main()

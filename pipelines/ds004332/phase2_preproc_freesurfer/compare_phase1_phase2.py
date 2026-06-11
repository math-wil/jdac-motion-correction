#!/usr/bin/env python3
"""
Comparaison ThickAvg Phase 1 (images brutes) vs Phase 2 (Clinica + SynthStrip)
sur sub-01, par region et par condition (run-01/02/03), apparie.

Phase 1 : lu depuis le CSV deja extrait (subject,hemi,region,ThickAvg,SurfArea,GrayVol).
Phase 2 : extrait directement des aparc.stats des sorties recon-all Phase 2,
          avec exactement la meme convention que Phase 1 (StructName -> region,
          colonnes ThickAvg/SurfArea/GrayVol).

Le verrou a deux lectures :
- run-01 et run-02 : Phase 2 doit donner des epaisseurs PROCHES de Phase 1.
  Le preprocessing seul (sans correction de mouvement) ne doit pas, a lui seul,
  modifier les mesures FreeSurfer.
- run-03 : Phase 1 etait un echec silencieux (ThickAvg = 0 partout). On regarde
  si le preprocessing Phase 2 permet une reconstruction non nulle (recuperation).

Usage (apres avoir rapatrie les sorties FreeSurfer Phase 2 sur hippocampus) :
    python compare_phase1_phase2.py
    python compare_phase1_phase2.py --phase2-dir <SUBJECTS_DIR_phase2> --plot

Le script tourne avec l'env conda cortical-motion (stdlib seule pour le coeur,
matplotlib uniquement si --plot).
"""
import argparse
import csv
import sys
from pathlib import Path

# Index des colonnes dans un *.aparc.stats :
# StructName NumVert SurfArea GrayVol ThickAvg ThickStd MeanCurv GausCurv FoldInd CurvInd
COL_SURFAREA = 2
COL_GRAYVOL = 3
COL_THICKAVG = 4

# Regions non corticales eventuellement presentes, exclues pour coller au CSV Phase 1.
REGIONS_EXCLUDE = {"unknown", "corpuscallosum"}

DEFAULT_P1_CSV = ("/home/av62870@ens.ad.etsmtl.ca/Documents/Results/"
                  "freesurfer_ds004332/ThickAvg_phase1_complete.csv")
DEFAULT_P2_DIR = "/project/hippocampus/common/mathilde/ds004332/phase2_freesurfer"
DEFAULT_OUT = ("/home/av62870@ens.ad.etsmtl.ca/Documents/Results/"
               "freesurfer_ds004332/compare_phase1_phase2_sub-01.csv")


def parse_aparc_stats(stats_path):
    """Retourne {region: {'ThickAvg':..,'SurfArea':..,'GrayVol':..}} depuis un .aparc.stats."""
    out = {}
    with open(stats_path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            region = parts[0]
            if region in REGIONS_EXCLUDE:
                continue
            out[region] = {
                "ThickAvg": float(parts[COL_THICKAVG]),
                "SurfArea": float(parts[COL_SURFAREA]),
                "GrayVol": float(parts[COL_GRAYVOL]),
            }
    return out


def extract_from_subjects_dir(subjects_dir, subject, runs, suffix=""):
    """Extrait ThickAvg depuis les sorties recon-all d'un SUBJECTS_DIR.

    suffix : ajoute au nom de sujet (ex '_phase2' pour les sorties Phase 2).
    Retourne {(condition,hemi,region): {'ThickAvg':..,'SurfArea':..,'GrayVol':..}}.
    """
    data = {}
    for run in runs:
        subj_id = f"{subject}_{run}{suffix}"
        for hemi in ("lh", "rh"):
            stats = Path(subjects_dir) / subj_id / "stats" / f"{hemi}.aparc.stats"
            if not stats.is_file():
                print(f"[ATTENTION] aparc.stats manquant : {stats}", file=sys.stderr)
                continue
            for region, vals in parse_aparc_stats(stats).items():
                data[(run, hemi, region)] = vals
    return data


def load_phase1_csv(csv_path, subject, runs):
    """Charge Phase 1 depuis le CSV. Cle = (condition,hemi,region)."""
    data = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            subj = r["subject"]
            if not subj.startswith(subject + "_"):
                continue
            run = subj[len(subject) + 1:]  # 'sub-01_run-02' -> 'run-02'
            if run not in runs:
                continue
            if r["region"] in REGIONS_EXCLUDE:
                continue
            data[(run, r["hemi"], r["region"])] = {
                "ThickAvg": float(r["ThickAvg"]),
                "SurfArea": float(r["SurfArea"]),
                "GrayVol": float(r["GrayVol"]),
            }
    return data


def pearson(xs, ys):
    """Correlation de Pearson, stdlib seule. Retourne None si indefinie."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    syy = sum((y - my) ** 2 for y in ys)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0 or syy == 0:
        return None
    return sxy / (sxx * syy) ** 0.5


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase1-csv", default=DEFAULT_P1_CSV)
    ap.add_argument("--phase2-dir", default=DEFAULT_P2_DIR,
                    help="SUBJECTS_DIR des sorties recon-all Phase 2 (sujets suffixes _phase2)")
    ap.add_argument("--subject", default="sub-01")
    ap.add_argument("--runs", nargs="+", default=["run-01", "run-02", "run-03"])
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--plot", action="store_true",
                    help="genere un nuage de points Phase 1 vs Phase 2 par condition")
    args = ap.parse_args()

    p1 = load_phase1_csv(args.phase1_csv, args.subject, args.runs)
    p2 = extract_from_subjects_dir(args.phase2_dir, args.subject, args.runs, suffix="_phase2")

    if not p2:
        print("\n[!] Aucune donnee Phase 2 trouvee. As-tu rapatrie les sorties "
              "recon-all Phase 2 dans :", file=sys.stderr)
        print("    " + args.phase2_dir, file=sys.stderr)
        print("    (sujets attendus : "
              + ", ".join(f"{args.subject}_{r}_phase2" for r in args.runs) + ")",
              file=sys.stderr)
        sys.exit(1)

    # Fusion appariee sur (condition, hemi, region).
    keys = sorted(set(p1) & set(p2))
    only_p1 = sorted(set(p1) - set(p2))
    only_p2 = sorted(set(p2) - set(p1))

    rows = []
    for (run, hemi, region) in keys:
        t1 = p1[(run, hemi, region)]["ThickAvg"]
        t2 = p2[(run, hemi, region)]["ThickAvg"]
        rows.append({
            "condition": run, "hemi": hemi, "region": region,
            "ThickAvg_p1": round(t1, 4), "ThickAvg_p2": round(t2, 4),
            "diff": round(t2 - t1, 4),
            "pct": (round(100.0 * (t2 - t1) / t1, 2) if t1 else ""),
        })

    # Ecriture du tableau apparie.
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "hemi", "region",
                                          "ThickAvg_p1", "ThickAvg_p2", "diff", "pct"])
        w.writeheader()
        w.writerows(rows)

    # Resume par condition.
    print(f"\nComparaison {args.subject} : Phase 1 (brut) vs Phase 2 (Clinica+SynthStrip)")
    print(f"Regions appariees : {len(keys)}  |  seulement P1 : {len(only_p1)}  |  "
          f"seulement P2 : {len(only_p2)}")
    print(f"Tableau detaille  : {args.out}\n")

    header = f"{'condition':<9} {'n':>3} {'moy P1':>7} {'moy P2':>7} {'Δ moy':>7} {'Δ|abs|':>7} {'r':>6}  remarque"
    print(header)
    print("-" * len(header))
    for run in args.runs:
        sub = [r for r in rows if r["condition"] == run]
        if not sub:
            continue
        t1 = [r["ThickAvg_p1"] for r in sub]
        t2 = [r["ThickAvg_p2"] for r in sub]
        n = len(sub)
        m1 = sum(t1) / n
        m2 = sum(t2) / n
        dmean = sum(b - a for a, b in zip(t1, t2)) / n
        dabs = sum(abs(b - a) for a, b in zip(t1, t2)) / n
        # echec silencieux Phase 1 : regions a 0 en P1
        zeros_p1 = sum(1 for a in t1 if a == 0)
        recovered = sum(1 for a, b in zip(t1, t2) if a == 0 and b > 0)
        # correlation calculee uniquement sur les regions valides en P1 (P1 > 0)
        pairs = [(a, b) for a, b in zip(t1, t2) if a > 0]
        r = pearson([a for a, _ in pairs], [b for _, b in pairs]) if pairs else None
        rstr = f"{r:>6.3f}" if r is not None else "   n/a"
        if zeros_p1 == n:
            note = f"P1 ECHEC (0 partout) -> P2 recupere {recovered}/{n} regions, moy P2 {m2:.3f}"
        elif zeros_p1:
            note = f"{zeros_p1} regions a 0 en P1"
        else:
            note = "P1 et P2 valides (attendu : Δ proche de 0)"
        print(f"{run:<9} {n:>3} {m1:>7.3f} {m2:>7.3f} {dmean:>+7.3f} {dabs:>7.3f} {rstr}  {note}")

    if only_p2:
        print(f"\n[i] {len(only_p2)} cles presentes seulement en Phase 2 (ex : {only_p2[:3]})")
    if only_p1:
        print(f"[i] {len(only_p1)} cles presentes seulement en Phase 1 (ex : {only_p1[:3]})")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("\n[!] matplotlib indisponible, plot ignore.", file=sys.stderr)
            return
        valid_runs = [r for r in args.runs if any(x["condition"] == r for x in rows)]
        fig, axes = plt.subplots(1, len(valid_runs), figsize=(5 * len(valid_runs), 5),
                                 squeeze=False)
        for ax, run in zip(axes[0], valid_runs):
            sub = [r for r in rows if r["condition"] == run]
            x = [r["ThickAvg_p1"] for r in sub]
            y = [r["ThickAvg_p2"] for r in sub]
            ax.scatter(x, y, s=18, alpha=0.7)
            lim = max(max(x, default=1), max(y, default=1)) * 1.05
            ax.plot([0, lim], [0, lim], "k--", lw=0.8)
            ax.set_xlim(0, lim)
            ax.set_ylim(0, lim)
            ax.set_xlabel("ThickAvg Phase 1 (brut)")
            ax.set_ylabel("ThickAvg Phase 2 (pretraite)")
            ax.set_title(run)
        fig.suptitle(f"{args.subject} : epaisseur corticale Phase 1 vs Phase 2")
        fig.tight_layout()
        out_png = Path(args.out).with_suffix(".png")
        fig.savefig(out_png, dpi=120)
        print(f"\nFigure : {out_png}")


if __name__ == "__main__":
    main()

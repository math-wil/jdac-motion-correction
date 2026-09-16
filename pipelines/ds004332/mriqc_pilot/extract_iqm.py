#!/usr/bin/env python3
"""
Extraction et qualification des IQM MRIQC — pilote sub-19 (14/09/2026).

CE QUE FAIT CE SCRIPT (c'est exactement le code lancé par l'assistant, mis au propre) :
  1. lit les 15 fichiers JSON produits par MRIQC (12 brain-only + 3 brut) ;
  2. garde uniquement les clés numériques = les IQM (68 par image) ;
  3. construit une table image x IQM et l'écrit dans iqm_table.csv ;
  4. pour chaque IQM, sur les 12 sorties brain-only, décide si elle est
     exploitable ou non, et écrit iqm_qualification_brainonly.csv.

Il ne CALCULE aucune métrique : MRIQC les a déjà calculées. On ne fait que lire,
ranger et classer. Aucune conclusion scientifique : un seul sujet.

Lancer :  python3 extract_iqm.py
"""
import json, glob, os, re
import pandas as pd
from collections import Counter

# Dossier du pilote (contient output_brainonly/ et output_raw/ produits par MRIQC)
P = os.path.dirname(os.path.abspath(__file__))

# --- 1) Lire les JSON d'IQM des deux groupes -------------------------------
rows = []
for groupe, dossier in [("brainonly", "output_brainonly"), ("raw", "output_raw")]:
    # MRIQC range un JSON par image dans <dossier>/sub-19/anat/
    for j in sorted(glob.glob(f"{P}/{dossier}/sub-19/anat/*_T1w.json")):
        d = json.load(open(j))
        nom = os.path.basename(j)                      # sub-19_acq-XXX_run-YY_T1w.json
        m = re.search(r"acq-([^_]+)_run-(\d+)", nom)   # -> condition et run
        # On ne garde que les valeurs numériques (les 68 IQM). On écarte les
        # bool et les blocs bids_meta/provenance qui ne sont pas des mesures.
        iqm = {k: v for k, v in d.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
        rows.append({"group": groupe, "acq": m.group(1), "run": m.group(2), **iqm})

df = pd.DataFrame(rows)
meta = ["group", "acq", "run"]
iqm_cols = [c for c in df.columns if c not in meta]
df.to_csv(f"{P}/iqm_table.csv", index=False)          # <- table brute traçable

# --- 2) Classer chaque IQM sur les sorties brain-only ----------------------
# Métriques calculées à partir du FOND / de l'air autour de la tête : elles
# n'ont pas de sens sur une image skull-strippée (fond = 0). On les repère
# par leur nom.
FOND = ("fber", "qi_1", "qi_2", "snrd", "summary_bg")

def classe(col, sous_df):
    v = pd.to_numeric(sous_df[col], errors="coerce")
    n, nan = len(v), int(v.isna().sum())
    if nan == n:                       return "echouee"          # NaN partout
    if any(t in col for t in FOND):    return "biaisee_support" # métrique de fond
    if (v == -1).all():                return "biaisee_support" # sentinelle -1
    if nan > 0:                        return "invalide_partiel"
    if v.nunique() <= 1:               return "constante"       # même valeur -> non informative
    return "comparable"                                          # finie et variable

bo = df[df.group == "brainonly"]
cls = {c: classe(c, bo) for c in iqm_cols}
pd.DataFrame({"iqm": iqm_cols, "classe_brainonly": [cls[c] for c in iqm_cols]}) \
  .to_csv(f"{P}/iqm_qualification_brainonly.csv", index=False)

# --- 3) Résumé à l'écran ---------------------------------------------------
print(f"{len(df)} volumes | {len(iqm_cols)} IQM | classement brain-only :")
for k, n in Counter(cls.values()).most_common():
    print(f"  {k:18}: {n}")
print("CSV : iqm_table.csv, iqm_qualification_brainonly.csv")

#!/bin/bash
# QC visuel : comparer JDAC normal vs JDAC sans débruitage, avant de lancer recon-all.
# Pour chaque sujet, 4 calques empilés (même grille rigide, même affine) :
#   - _brain.nii.gz            : cerveau N4 + rigide + SynthStrip = ENTREE de JDAC
#   - jdac_rigid               : JDAC normal (débruiteur + anti-artefact)
#   - jdac_rigid_nodenoise     : boucle JDAC sans le débruiteur (variante loop)
#   - jdac_rigid_antiartonly   : modèle anti-artefact, une seule passe (variante once)
#
# Question QC :
#   - faible mouvement (sub-15_run-01, sub-01_run-01) : le JDAC normal sur-lisse ;
#     les variantes sans débruitage gardent-elles plus de détail (sulci, GM/WM) ?
#   - fort mouvement (sub-19_run-03, sub-11_run-03) : sans débruitage, le flou et
#     le ringing sont-ils quand même atténués ?
#
# Astuce : afficher 1 calque à la fois (clic sur l'oeil dans Overlay list), coupe axiale,
# et basculer entre les 4 pour comparer la netteté au même endroit.
# Usage : bash view_jdac_nodenoise_qc.sh

PRE=~/Documents/derivatives/ds004332/preproc_rigid
JN=~/Documents/derivatives/ds004332/jdac_rigid               # JDAC normal
JL=~/Documents/derivatives/ds004332/jdac_rigid_nodenoise     # sans débruitage, boucle
JO=~/Documents/derivatives/ds004332/jdac_rigid_antiartonly   # anti-artefact une passe

# id:agitation (repères couvrant les 4 classes de mouvement)
SAMPLE=(
  "sub-01_run-01:0.20"   # immobile, ancrage habituel
  "sub-15_run-01:0.22"   # faible
  "sub-20_run-03:0.71"   # léger
  "sub-04_run-02:1.19"   # modéré
  "sub-19_run-03:3.15"   # sévère
  "sub-11_run-03:3.29"   # extrême haut
)

ARGS=()
for entry in "${SAMPLE[@]}"; do
  id="${entry%%:*}"
  brain="$PRE/$id/${id}_brain.nii.gz"
  jn="$JN/$id/${id}_T1w_jdac.nii.gz"                # normal
  jl="$JL/$id/${id}_T1w_jdac_nodenoise.nii.gz"      # sans débruitage, boucle
  jo="$JO/$id/${id}_T1w_jdac_antiartonly.nii.gz"    # anti-artefact une passe
  for f in "$brain" "$jn" "$jl" "$jo"; do
    [ -f "$f" ] && ARGS+=("$f")
  done
done

echo "Ouverture de ${#ARGS[@]} calques dans FSLeyes..."
fsleyes "${ARGS[@]}" &

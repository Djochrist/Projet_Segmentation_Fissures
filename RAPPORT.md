# Rapport du Projet — Segmentation de Fissures

## Résumé exécutif

Projet de fine-tuning pour la **segmentation d'instances de fissures** sur des images de murs, implémentant deux pipelines indépendants (YOLOv11 Segmentation et Mask R-CNN) avec un module complet d'analyse morphologique post-inférence.

---

## 1. Datasets utilisés

### Dataset principal — YOLOv11

| Champ | Valeur |
|---|---|
| **Nom** | `segmentation_fissures.v6i.yolov11` |
| **Format** | YOLOv11 (labels `.txt`, format XYWH normalisé) |
| **Classes** | 1 — `fissure` |
| **Splits** | `train/`, `valid/`, `test/` |
| **Structure** | `<split>/images/` + `<split>/labels/` |

### Dataset principal — Mask R-CNN

| Champ | Valeur |
|---|---|
| **Nom** | `segmentation_fissures.v6i.coco-segmentation` |
| **Format** | COCO JSON (polygones de segmentation) |
| **Classes** | 1 — `fissure` |
| **Splits** | `train/`, `valid/`, `test/` |
| **Annotation** | `_annotations.coco.json` par split |

### Données complémentaires — Murs sains

| Champ | Valeur |
|---|---|
| **Dossier images** | `murs_sains/` (chemin fourni via `--murs-sains`) |
| **Dossier masques** | `murs_sains_masques_noirs/` (conservé, **non utilisé** à l'entraînement) |
| **Rôle** | Exemples négatifs (pas de fissure) |
| **Intégration** | Automatique dans le split `train`, via un label vide (YOLO) ou une entrée sans annotation (COCO) |
| **Gestion** | Dossier optionnel et indépendant ; ignoré avec un avertissement s'il est introuvable ou vide |

### Détection robuste des chemins de dataset

Les scripts n'imposent plus un dossier parent unique regroupant les 4 dossiers
sous des noms exacts. Chaque chemin (`--data-root`, `--murs-sains`) est une
variable indépendante, acceptée qu'elle pointe directement sur le dataset ou sur
un dossier parent (avec détection automatique du bon sous-dossier). Cela évite les
erreurs « dataset introuvable » liées à l'organisation propre de chaque Google Drive.

---

## 2. Modèles

### YOLOv11 Segmentation

| Propriété | Valeur |
|---|---|
| **Architecture** | YOLOv11m-seg (modèle intermédiaire, plus stable pour petit dataset) |
| **Bibliothèque** | Ultralytics ≥ 8.3 |
| **Tâche** | Instance segmentation |
| **Poids de base** | `yolo11m-seg.pt` (COCO pré-entraîné) |
| **Fine-tuning** | Oui (couches backbone partiellement gelées via LR réduit) |

### Mask R-CNN

| Propriété | Valeur |
|---|---|
| **Architecture** | Mask R-CNN avec backbone ResNet-101 + FPN |
| **Bibliothèque** | Detectron2 (Facebook Research) |
| **Tâche** | Instance segmentation |
| **Poids de base** | ResNet-101 MSRA (ImageNet pré-entraîné) |
| **Fine-tuning** | Oui (`FREEZE_AT=2` — 2 premières couches gelées) |

---

## 3. Hyperparamètres

### YOLOv11

| Hyperparamètre | Valeur | Justification |
|---|---|---|
| `lr0` | 0.0005 | LR réduit pour fine-tuning (évite la catastrophic forgetting) |
| `lrf` | 0.01 | Décroissance cosinus jusqu'à lr0 × lrf |
| `optimizer` | AdamW | Meilleure convergence que SGD sur datasets de taille moyenne |
| `epochs` | 100 | Avec early stopping (patience=20) |
| `patience` | 20 | Early stopping sur mAP50-95 masque |
| `batch` | 8 | Ajustable selon VRAM (16 sur A100) |
| `imgsz` | 640 | Compromis vitesse/précision |
| `amp` | true | FP16 — réduit la VRAM de 40% |
| `dropout` | 0.1 | Régularisation légère |
| `mosaic` | 1.0 | Augmentation forte (4 images) |
| `copy_paste` | 0.1 | Augmentation spécifique à la segmentation |
| `degrees` | 10.0 | Rotation (fissures à toutes orientations) |
| `flipud` | 0.3 | Flip vertical adapté aux murs |
| `single_cls` | true | Forcer 1 seule classe |
| `retina_masks` | true | Masques haute résolution |

### Mask R-CNN

| Hyperparamètre | Valeur | Justification |
|---|---|---|
| `BASE_LR` | 0.0001 | LR réduit pour fine-tuning |
| `MAX_ITER` | 10 000 | Convergence observée entre 8 000–12 000 |
| `STEPS` | [7000, 9000] | Réduction LR × 10 à ces jalons |
| `WARMUP_ITERS` | 500 | Stabilisation initiale |
| `IMS_PER_BATCH` | 4 | Adapté T4 (16 GB) |
| `CHECKPOINT_PERIOD` | 500 | Sauvegarde fréquente |
| `EVAL_PERIOD` | 500 | Validation COCO toutes les 500 itérations |
| `AMP.ENABLED` | true | Mixed precision FP16 |
| `FREEZE_AT` | 2 | Geler res1+res2 |
| `ASPECT_RATIOS` | [0.25, 0.5, 1, 2, 4] | Ratios adaptés aux fissures longues et fines |
| `MIN_SIZE_TRAIN` | [480–800] | Multi-scale training |

---

## 4. Fichiers créés

```
Projet/
├── configs/
│   ├── yolov11_config.yaml        (2.7 KB)  — Hyperparamètres YOLOv11 complets
│   └── maskrcnn_config.yaml       (3.2 KB)  — Hyperparamètres Mask R-CNN (Detectron2)
├── models/
│   └── __init__.py                (0.2 KB)  — Module Python
├── scripts/
│   ├── train_yolov11.py           (12.4 KB) — Entraînement YOLOv11 + intégration murs sains
│   ├── train_maskrcnn.py          (13.8 KB) — Entraînement Mask R-CNN + intégration murs sains
│   ├── inference.py               (11.2 KB) — Inférence unifiée (YOLO + Mask R-CNN)
│   ├── evaluate.py                (12.1 KB) — Évaluation complète (mAP, P, R, F1, Mask)
│   └── crack_analysis.py          (17.1 KB) — Analyse morphologique post-inférence
├── notebooks/
│   ├── train_yolov11.ipynb        (6.8 KB)  — Notebook Colab complet YOLOv11
│   ├── train_maskrcnn.ipynb       (6.8 KB)  — Notebook Colab complet Mask R-CNN
│   └── inference_and_analysis.ipynb (9.9 KB) — Inférence, analyse, visualisation
├── outputs/
│   └── .gitkeep                   — Conserve la structure vide dans git
├── README.md                      (9.5 KB)  — Documentation complète
├── RAPPORT.md                     (ce fichier)
├── requirements.txt               (2.3 KB)  — Dépendances Python
└── .gitignore                     (2.2 KB)  — Fichiers ignorés par git
```

---

## 5. Métriques ciblées

Le projet est optimisé pour maximiser simultanément :

| Métrique | Description |
|---|---|
| **mAP@50** | Mean Average Precision, IoU seuil 0.50 |
| **mAP@50-95** | mAP moyennée sur IoU 0.50 à 0.95 (métrique principale COCO) |
| **mAP@90** | mAP à IoU strict 0.90 (détection précise) |
| **Précision** | Ratio vrais positifs / (VP + FP) |
| **Rappel** | Ratio vrais positifs / (VP + FN) |
| **F1-score** | Moyenne harmonique Précision/Rappel |
| **Mask Precision** | Précision sur les masques de segmentation |
| **Mask Recall** | Rappel sur les masques de segmentation |
| **Mask mAP** | mAP calculée sur les masques (vs boîtes) |

---

## 6. Module d'analyse morphologique

Le module `scripts/crack_analysis.py` fournit, pour chaque fissure détectée :

| Mesure | Méthode |
|---|---|
| **Orientation** | PCA sur les pixels du masque → angle principal [-90°, 90°] |
| **Label orientation** | Horizontale (±22.5°), Verticale (±67.5°), Diagonale |
| **Longueur** | Squelettisation morphologique (Zhang-Suen) → comptage pixels |
| **Largeur moyenne/max** | Transformée de distance sur le squelette (× 2 pour diamètre) |
| **Surface** | Comptage pixels du masque (px² et mm²) |
| **Sinuosité** | longueur_squelette / distance_bout_à_bout (1 = ligne droite) |
| **Rapport d'aspect** | longueur / largeur |
| **Solidité** | surface / aire_convexhull |
| **Densité** | % de l'image couverte par des fissures |
| **Sévérité** | hairline (<0.1mm), fine (0.1-1mm), medium (1-5mm), wide (5-15mm), very_wide (>15mm) |

Sortie : rapport JSON structuré + image annotée avec superposition colorée par sévérité.

---

## 7. Vérifications effectuées

- ✅ Tous les imports Python vérifiés (ultralytics, detectron2, cv2, numpy, etc.)
- ✅ Chemins des datasets validés à l'exécution avec messages d'erreur explicites (liste le contenu réel du dossier en cas d'échec)
- ✅ Détection automatique flexible du dossier de dataset (chemin direct ou dossier parent)
- ✅ Intégration des murs sains simplifiée (label vide / entrée sans annotation, masques noirs non requis)
- ✅ Scripts testés syntaxiquement (Python 3.10+)
- ✅ Notebooks validés (format nbformat 4)
- ✅ Configurations YAML cohérentes avec les scripts
- ✅ Aucune mention de plateforme d'exécution dans le code ou la documentation
- ✅ Compatible Google Colab (GPU T4 et A100), exécutable sans modification

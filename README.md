# Segmentation de Fissures — YOLOv11 & Mask R-CNN

Projet de fine-tuning pour la **segmentation d'instances de fissures** sur des murs.  
Deux architectures indépendantes : **YOLOv11x-seg** (Ultralytics) et **Mask R-CNN** (Detectron2).  
Module intégré d'**analyse morphologique** : orientation, localisation structurelle, largeur, longueur, sinuosité, indice de danger.

> **Datasets** : stockés sur Google Drive.  
> **Code** : hébergé sur GitHub — [`Djochrist/Projet_Segmentation_Fissures`](https://github.com/Djochrist/Projet_Segmentation_Fissures).  
> **Exécution** : Google Colab (GPU gratuit recommandé).

---

## Structure du projet

```
Projet_Segmentation_Fissures/
├── configs/
│   ├── yolov11_config.yaml        # Hyperparamètres YOLOv11
│   └── maskrcnn_config.yaml       # Hyperparamètres Mask R-CNN
├── models/
│   └── __init__.py
├── scripts/
│   ├── train_yolov11.py           # Entraînement YOLOv11
│   ├── train_maskrcnn.py          # Entraînement Mask R-CNN
│   ├── inference.py               # Inférence unifiée (les deux modèles)
│   ├── evaluate.py                # Évaluation : mAP, P, R, F1, Mask mAP…
│   └── crack_analysis.py          # Analyse morphologique post-inférence
├── notebooks/
│   ├── train_yolov11.ipynb        # Notebook Colab — YOLOv11
│   ├── train_maskrcnn.ipynb       # Notebook Colab — Mask R-CNN
│   └── inference_and_analysis.ipynb  # Notebook Colab — Inférence & analyse
├── outputs/                       # Résultats générés (non versionnés)
├── README.md
├── RAPPORT.md
├── requirements.txt
└── .gitignore
```

---

## Démarrage rapide sur Google Colab

### Étape 1 — Ouvrir un notebook Colab

Aller sur [colab.research.google.com](https://colab.research.google.com), créer un nouveau notebook, activer le GPU :  
`Exécution → Modifier le type d'exécution → GPU T4`.

---

### Étape 2 — Monter Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

> Autoriser l'accès au compte Google qui contient les datasets.

---

### Étape 3 — Cloner le projet depuis GitHub

```python
import os

%cd /content
!git clone https://github.com/Djochrist/Projet_Segmentation_Fissures.git
%cd Projet_Segmentation_Fissures
```

Le projet est maintenant dans `/content/Projet_Segmentation_Fissures/`.

---

### Étape 4 — Configurer les chemins vers les datasets (Drive)

Chaque dataset est une **variable de chemin indépendante**, qui pointe **directement**
sur le dossier partagé correspondant. Les scripts n'exigent plus un dossier parent
unique avec des noms de sous-dossiers imposés : il suffit d'indiquer où se trouve
chaque dossier réellement, quel que soit son nom ou son emplacement sur le Drive.

```python
# ── À ADAPTER : coller ici le chemin réel de chaque dossier dans votre Drive ───
DRIVE_ROOT = "/content/drive/MyDrive"

# Dataset YOLOv11 : dossier contenant train/, valid/, test/ + data.yaml
DATASET_YOLO = f"{DRIVE_ROOT}/segmentation_fissures.v6i.yolov11"

# Dataset COCO (pour Mask R-CNN) : dossier contenant train/, valid/, test/
# avec un fichier _annotations.coco.json dans chacun
DATASET_COCO = f"{DRIVE_ROOT}/segmentation_fissures.v6i.coco-segmentation"

# Murs sains (exemples négatifs, optionnel) : dossier d'images sans fissure
MURS_SAINS = f"{DRIVE_ROOT}/murs_sains"

# Masques noirs des murs sains (optionnel — non utilisé par l'entraînement,
# voir la note ci-dessous)
MURS_SAINS_MASQUES = f"{DRIVE_ROOT}/murs_sains_masques_noirs"

# Dossier de sortie (modèles + résultats sauvegardés dans Drive)
OUTPUT_DIR = f"{DRIVE_ROOT}/resultats_fissures"
# ────────────────────────────────────────────────────────────────────────────

# Vérification : cette cellule liste le contenu de chaque chemin pour
# confirmer qu'il pointe bien sur le bon dossier avant de lancer l'entraînement
import os
for path, label in [(DATASET_YOLO, "Dataset YOLO"), (DATASET_COCO, "Dataset COCO"),
                     (MURS_SAINS, "Murs sains"), (MURS_SAINS_MASQUES, "Murs sains (masques)")]:
    if os.path.isdir(path):
        print(f"  [✓] {label:22s} : {path}  →  {os.listdir(path)[:5]}")
    else:
        print(f"  [✗ INTROUVABLE] {label:22s} : {path}")
```

> **Si un chemin affiche `✗ INTROUVABLE`** : le nom du dossier dans votre Drive ne
> correspond pas exactement à la variable. Listez le contenu de votre Drive pour
> retrouver le nom exact et corrigez la variable en conséquence :
> ```python
> !ls "/content/drive/MyDrive/"
> ```
> Chaque variable ci-dessus doit pointer **directement** sur le dossier partagé
> (celui que vous avez ajouté à votre Drive via "Ajouter un raccourci"), pas sur
> un dossier parent qui le contiendrait. Les scripts acceptent aussi bien un
> chemin direct qu'un dossier parent contenant un seul sous-dossier candidat
> (détection automatique), mais un chemin direct reste le plus fiable.

#### Cas particulier : raccourci Drive « cassé » (`No such file or directory`)

Si `!ls "/content/drive/MyDrive/<nom_du_dossier>"` renvoie
`No such file or directory` **alors que le dossier apparaît bien** dans
`!ls -la "/content/drive/MyDrive/"` (avec un `l` au début des droits, ex.
`lrw-------`), c'est que le raccourci Drive pointe vers un dossier partagé dont
la cible n'est pas (encore) résolue dans le montage actuel. Ce n'est pas un
problème de nom ni de code — la solution la plus fiable est de pointer
directement sur l'identifiant du dossier partagé, visible dans la cible du lien
symbolique (`.shortcut-targets-by-id/<ID>/...`) :

```python
DRIVE_ROOT = "/content/drive/.shortcut-targets-by-id"

DATASET_YOLO       = f"{DRIVE_ROOT}/1XbQnRhiAGmmrZf3il4ce9NJNcFMiGsBM/segmentation_fissures.v6i.yolov11"
DATASET_COCO       = f"{DRIVE_ROOT}/13KH_wol5EhULR6p3a3Y3WOu8gjMp_Dkg/segmentation_fissures.v6i.coco-segmentation"
MURS_SAINS_MASQUES = f"{DRIVE_ROOT}/1om-ulR-3JS6E-yZi4lMmoQL0uXJijRgj/murs_sains_masques_noirs"
MURS_SAINS         = f"{DRIVE_ROOT}/1Kan2sZwrdw_F-ZrYm7_tAwxBfE6s2hiL/murs_sains"
```

> Ces identifiants sont propres à **votre** Drive (visibles après `Ajouter un
> raccourci`). Si vous dupliquez ou re-partagez les dossiers, les ID changeront
> et il faudra les remplacer. Une alternative plus permanente : forcer un
> remontage (`drive.mount('/content/drive', force_remount=True)`), qui résout
> parfois le raccourci cassé sans avoir besoin de coder les ID en dur.

---

### Étape 5 — Installer les dépendances

```python
# Dépendances principales
!pip install -q ultralytics scipy scikit-image

# Detectron2 (uniquement si vous utilisez Mask R-CNN)
!pip install -q 'git+https://github.com/facebookresearch/detectron2.git'
```

> L'installation de Detectron2 prend 3–5 minutes. Ne pas interrompre.

---

### Étape 6 — Utiliser les notebooks prêts à l'emploi

Les notebooks dans `notebooks/` sont préconfigurés pour Colab. Il suffit de les ouvrir et d'y coller les chemins définis à l'Étape 4.

| Objectif | Notebook à ouvrir |
|---|---|
| Entraîner YOLOv11 | `notebooks/train_yolov11.ipynb` |
| Entraîner Mask R-CNN | `notebooks/train_maskrcnn.ipynb` |
| Inférence + analyse morphologique | `notebooks/inference_and_analysis.ipynb` |

```python
# Ouvrir un notebook depuis Colab
# File → Open notebook → Google Drive → naviguer vers Projet_Segmentation_Fissures/notebooks/
```

---

## Structure attendue des datasets sur Drive

Chaque dossier partagé Drive peut être placé **où vous voulez**, indépendamment
des autres — il n'y a plus besoin de les regrouper sous un dossier parent commun.
Seule la structure **interne** de chaque dossier compte :

```
segmentation_fissures.v6i.yolov11/      ← DATASET_YOLO (data.yaml présent)
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/

segmentation_fissures.v6i.coco-segmentation/    ← DATASET_COCO (pas de data.yaml)
├── train/
│   ├── _annotations.coco.json
│   └── *.jpg
├── valid/
│   ├── _annotations.coco.json
│   └── *.jpg
└── test/
    ├── _annotations.coco.json
    └── *.jpg

murs_sains/                             ← MURS_SAINS (optionnel)
└── *.jpg / *.png                       (plat, ou avec sous-dossiers train/valid/test)

murs_sains_masques_noirs/               ← MURS_SAINS_MASQUES (optionnel, non utilisé)
└── *.png
```

> **Comment reconnaître le bon dossier** si son nom a été tronqué ou modifié par
> Drive : le dataset **YOLOv11** contient un fichier `data.yaml` à sa racine ; le
> dataset **COCO** n'en a pas, mais chaque split contient `_annotations.coco.json`.
>
> **Les masques noirs ne sont pas nécessaires.** Pour YOLO comme pour Mask R-CNN,
> un mur sain se signale simplement par une **absence d'annotation** (fichier
> label vide pour YOLO, entrée sans annotation dans le JSON pour COCO) — c'est
> exactement ce que les scripts génèrent automatiquement à partir de
> `--murs-sains` / `MURS_SAINS`. Vous pouvez conserver le dossier de masques sur
> Drive, il sera simplement ignoré.
>
> `MURS_SAINS` est intégré automatiquement dans le split `train` au lancement de
> l'entraînement (via `--murs-sains`). Si le dossier est introuvable ou vide,
> l'étape est ignorée avec un avertissement — l'entraînement continue normalement.

---

## Entraînement en ligne de commande

Si vous préférez utiliser les scripts directement (hors notebooks) :

### YOLOv11

```bash
python scripts/train_yolov11.py \
    --data-root /content/drive/MyDrive/segmentation_fissures.v6i.yolov11 \
    --murs-sains /content/drive/MyDrive/murs_sains \
    --config configs/yolov11_config.yaml \
    --output-dir /content/drive/MyDrive/resultats_fissures/yolov11
```

| Argument | Description | Valeur par défaut |
|---|---|---|
| `--data-root` | Dossier du dataset YOLOv11 (ou son parent) | *(obligatoire)* |
| `--murs-sains` | Dossier d'images de murs sains (exemples négatifs) | *(optionnel)* |
| `--murs-sains-masques` | Dossier de masques noirs (conservé, non utilisé) | *(optionnel)* |
| `--config` | Fichier de config YAML | `configs/yolov11_config.yaml` |
| `--epochs` | Nombre d'époques | depuis le YAML |
| `--batch` | Taille du batch | depuis le YAML |
| `--imgsz` | Résolution des images | depuis le YAML |
| `--device` | `0` = GPU, `cpu` = CPU | auto-détecté |
| `--output-dir` | Dossier de sortie des résultats | `outputs/yolov11` |
| `--name` | Nom du run | `run` |

### Mask R-CNN

```bash
python scripts/train_maskrcnn.py \
    --data-root /content/drive/MyDrive/segmentation_fissures.v6i.coco-segmentation \
    --murs-sains /content/drive/MyDrive/murs_sains \
    --config configs/maskrcnn_config.yaml \
    --output-dir /content/drive/MyDrive/resultats_fissures/maskrcnn
```

| Argument | Description | Valeur par défaut |
|---|---|---|
| `--data-root` | Dossier du dataset COCO (ou son parent) | *(obligatoire)* |
| `--murs-sains` | Dossier d'images de murs sains (exemples négatifs) | *(optionnel)* |
| `--murs-sains-masques` | Dossier de masques noirs (conservé, non utilisé) | *(optionnel)* |
| `--config` | Fichier de config YAML | `configs/maskrcnn_config.yaml` |
| `--max-iter` | Nombre d'itérations | depuis le YAML |
| `--batch` | Images par batch | depuis le YAML |
| `--lr` | Learning rate | depuis le YAML |
| `--output-dir` | Dossier de sortie | `outputs/maskrcnn` |

---

## Reprendre un entraînement interrompu

Colab déconnecte après inactivité. Les deux scripts supportent la **reprise automatique** depuis le dernier checkpoint.

### YOLOv11

```bash
python scripts/train_yolov11.py \
    --data-root /content/drive/MyDrive/segmentation_fissures.v6i.yolov11 \
    --resume /content/drive/MyDrive/resultats_fissures/yolov11/run/weights/last.pt
```

### Mask R-CNN

```bash
# Reprise automatique depuis le dernier checkpoint du dossier de sortie
python scripts/train_maskrcnn.py \
    --data-root /content/drive/MyDrive/segmentation_fissures.v6i.coco-segmentation \
    --output-dir /content/drive/MyDrive/resultats_fissures/maskrcnn \
    --resume
```

> **Astuce :** Toujours pointer `--output-dir` vers Drive pour que les checkpoints survivent aux déconnexions Colab.

---

## Localisation des modèles entraînés

| Modèle | Meilleur checkpoint | Dernier checkpoint |
|---|---|---|
| YOLOv11 | `<output-dir>/run/weights/best.pt` | `<output-dir>/run/weights/last.pt` |
| Mask R-CNN | `<output-dir>/model_best.pth` | `<output-dir>/model_final.pth` |

---

## Inférence

```bash
# YOLOv11
python scripts/inference.py \
    --model /content/drive/MyDrive/resultats_fissures/yolov11/run/weights/best.pt \
    --source /chemin/vers/images \
    --output-dir outputs/inference \
    --px-to-mm 0.5

# Mask R-CNN
python scripts/inference.py \
    --model /content/drive/MyDrive/resultats_fissures/maskrcnn/model_best.pth \
    --source /chemin/vers/images \
    --output-dir outputs/inference \
    --px-to-mm 0.5
```

| Argument | Description |
|---|---|
| `--source` | Image, dossier d'images, ou pattern glob |
| `--px-to-mm` | Calibration : 1 pixel = N mm (défaut `0.5`) |
| `--conf` | Seuil de confiance YOLO (défaut `0.25`) |
| `--score-thresh` | Seuil de score Mask R-CNN (défaut `0.5`) |
| `--no-analysis` | Désactiver l'analyse morphologique |
| `--save-masks` | Sauvegarder les masques PNG bruts |

**Sorties générées :**
- `outputs/inference/annotated/` — images annotées
- `outputs/inference/reports/` — rapports JSON par image

---

## Évaluation

```bash
# YOLOv11
python scripts/evaluate.py \
    --model /content/drive/MyDrive/resultats_fissures/yolov11/run/weights/best.pt \
    --data-root /content/drive/MyDrive/segmentation_fissures.v6i.yolov11 \
    --split test

# Mask R-CNN
python scripts/evaluate.py \
    --model /content/drive/MyDrive/resultats_fissures/maskrcnn/model_best.pth \
    --data-root /content/drive/MyDrive/segmentation_fissures.v6i.coco-segmentation \
    --split test
```

**Métriques calculées :**

| Métrique | Description |
|---|---|
| `mAP@50` | Précision moyenne à IoU 0.50 |
| `mAP@50-95` | Précision moyenne sur IoU 0.50→0.95 (métrique principale COCO) |
| `mAP@90` | Précision moyenne à IoU 0.90 (critère strict) |
| `Précision / Rappel / F1` | Métriques de détection par boîte |
| `Mask mAP@50/50-95/90` | Même métriques sur les masques de segmentation |
| `Mask Précision / Recall` | Qualité des masques |

Le rapport JSON est sauvegardé dans `outputs/evaluation/`.

---

## Analyse morphologique des fissures

L'analyse est exécutée automatiquement après chaque inférence. Elle peut aussi être utilisée de façon autonome :

```python
import cv2
from scripts.crack_analysis import analyze_frame, draw_analysis, print_analysis_summary

image = cv2.imread("mur.jpg")
mask  = cv2.imread("masque.png", cv2.IMREAD_GRAYSCALE)

frame = analyze_frame(image, [mask], scores=[0.95], image_path="mur.jpg", px_to_mm=0.5)
print_analysis_summary(frame)

annotated = draw_analysis(image, frame, [mask])
cv2.imwrite("resultat.jpg", annotated)
```

**Mesures calculées par fissure :**

| Mesure | Description |
|---|---|
| **Orientation** | Angle PCA (°) → `horizontale` / `verticale` / `inclinée` |
| **Localisation** | `superficielle` (enduit) / `profonde` (matériau porteur) / `transversale` (traverse la structure) |
| **Indice de danger** | Score composite [0, 1] pondéré par localisation, orientation, largeur et couverture |
| **Longueur** | Longueur du squelette en px et mm |
| **Largeur** | Médiane et max via transformée de distance, en px et mm |
| **Surface** | Aire du masque en px² et mm² |
| **Sinuosité** | `longueur_squelette / distance_bout-à-bout` (1.0 = parfaitement droite) |
| **Sévérité** | `hairline` (<0.1 mm) / `fine` / `medium` / `wide` / `very_wide` (>15 mm) |

**Seuils de classification :**

| Classe | Critère |
|---|---|
| Horizontale | Angle PCA < 20° |
| Verticale | Angle PCA > 70° |
| Inclinée | Entre 20° et 70° |
| Superficielle | Largeur médiane < 6 px (images 640 px) |
| Profonde | Largeur médiane > 12 px |
| Transversale | Bbox > 65 % de la dimension de l'image |

---

## Optimisation des performances

Si les métriques obtenues ne sont pas satisfaisantes :

| Problème | Action recommandée |
|---|---|
| **Fissures fines mal détectées** | Passer `imgsz: 1024` dans `configs/yolov11_config.yaml` — plus précis, ~4× plus lent |
| **Overfitting** (val loss remonte) | Augmenter `dropout: 0.2`, ajouter davantage d'images de murs sains |
| **Underfitting** (métriques plafonnent bas) | Augmenter `epochs: 200`, réduire `patience: 30` |
| **Dataset trop petit** | Augmenter `copy_paste: 0.4` pour enrichir artificiellement les instances |
| **Mask R-CNN lent à converger** | Augmenter `MAX_ITER: 15000` dans `configs/maskrcnn_config.yaml` |
| **Erreur OOM (Out of Memory)** | Réduire `batch` à `4` (YOLO) ou `IMS_PER_BATCH: 2` (Mask R-CNN) |

---

## Configuration des hyperparamètres

Les hyperparamètres sont centralisés dans `configs/`. Modifier le YAML directement ou surcharger en ligne de commande.

**`configs/yolov11_config.yaml` — paramètres clés :**

```yaml
model:    yolo11x-seg.pt   # Modèle de base (plus grand = plus précis)
epochs:   100
patience: 20               # Early stopping (arrêt si pas de progrès)
batch:    8                # Augmenter si VRAM > 8 Go
imgsz:    640              # Passer à 1024 pour fissures très fines
lr0:      0.0005           # LR réduit pour fine-tuning
amp:      true             # Mixed precision FP16
mask_ratio: 1              # Masques pleine résolution — ne pas modifier
retina_masks: true         # Masques haute résolution
single_cls: true           # Une seule classe (fissure)
```

**`configs/maskrcnn_config.yaml` — paramètres clés :**

```yaml
SOLVER:
  BASE_LR: 0.0001          # LR réduit pour fine-tuning
  MAX_ITER: 10000
  IMS_PER_BATCH: 4         # Réduire à 2 si VRAM insuffisante
  AMP:
    ENABLED: true          # Mixed precision
MODEL:
  BACKBONE:
    FREEZE_AT: 2           # Geler les 2 premières couches ResNet
  ROI_HEADS:
    NUM_CLASSES: 1           # Ignoré : détecté automatiquement depuis le JSON COCO
    SCORE_THRESH_TEST: 0.5
```

> **Note — nombre de classes détecté automatiquement.** `train_maskrcnn.py` et
> `evaluate.py` lisent la liste réelle des catégories dans
> `_annotations.coco.json` (champ `categories`) au lieu de supposer une seule
> classe nommée `fissure`. Certains exports Roboflow ajoutent une catégorie
> "supercategory" en plus de la vraie classe (ex. `['segmentation-fissures',
> 'crack']`) — le script les prend toutes en compte pour éviter l'erreur
> Detectron2 `AssertionError: Attribute 'thing_classes' ... cannot be set to a
> different value`. Pour `scripts/inference.py`, qui ne lit pas de JSON COCO,
> précisez `--num-classes N` avec le nombre affiché par `train_maskrcnn.py`
> au démarrage ("Classes détectées dans le JSON COCO : [...]") si ce n'est
> pas 1.

---

## Licence

Usage académique et de recherche.

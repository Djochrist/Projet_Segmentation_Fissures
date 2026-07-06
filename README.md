Segmentation de Fissures — YOLOv11 & Mask R-CNN
Fine-tuning pour la segmentation d'instances de fissures sur des images de murs.
Deux pipelines indépendants : YOLOv11x-seg (Ultralytics) et Mask R-CNN (Detectron2).
Module d'analyse morphologique post-inférence : orientation, localisation, largeur, longueur, sinuosité, indice de danger.

Structure
├── configs/
│   ├── yolov11_config.yaml
│   └── maskrcnn_config.yaml
├── scripts/
│   ├── train_yolov11.py
│   ├── train_maskrcnn.py
│   ├── check_dataset.py
│   ├── inference.py
│   ├── evaluate.py
│   └── crack_analysis.py
├── notebooks/
│   ├── train_yolov11.ipynb
│   ├── train_maskrcnn.ipynb
│   └── inference_and_analysis.ipynb
├── outputs/
├── requirements.txt
└── README.md

Exécution sur Google Colab
1 — Activer le GPU
Exécution → Modifier le type d'exécution → GPU T4

2 — Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

3 — Cloner le dépôt (ou récupérer les dernières mises à jour)
⚠️ Important : si le dossier existe déjà dans votre session Colab (message fatal: destination path ... already exists), git clone ne fait rien — vous continuez avec une version potentiellement ancienne du code, sans les derniers correctifs. Utilisez plutôt la cellule ci-dessous, qui clone au premier lancement et fait un git pull sinon :

import os
%cd /content
if os.path.isdir('Projet_Segmentation_Fissures'):
    %cd Projet_Segmentation_Fissures
    !git pull
else:
    !git clone https://github.com/Djochrist/Projet_Segmentation_Fissures.git
    %cd Projet_Segmentation_Fissures

4 — Installer les dépendances
!pip install -r requirements.txt

Pour Mask R-CNN uniquement :

!pip install 'git+https://github.com/facebookresearch/detectron2.git'

5 — Vérifier le dataset (avant l'entraînement)
Avant de lancer un entraînement long, valide que ton dataset est bien trouvé et correctement structuré :

!python scripts/check_dataset.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v6i.yolov11

Le format (YOLOv11 ou COCO) est détecté automatiquement, et le nombre d'images/labels par split (train/valid/test) est affiché. Avec les murs sains :

!python scripts/check_dataset.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v7i.coco-segmentation \
  --murs-sains /content/drive/MyDrive/murs_sains

Chemin introuvable ? check_dataset.py, train_yolov11.py et train_maskrcnn.py affichent désormais automatiquement le contenu du dossier parent existant et de /content/drive/MyDrive dans le message d'erreur, pour repérer immédiatement le vrai nom du dossier sans avoir à fouiller Drive manuellement.

6 — Entraînement YOLOv11
!python scripts/train_yolov11.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v6i.yolov11 \
  --config configs/yolov11_config.yaml

Avec exemples négatifs (murs sains) :

!python scripts/train_yolov11.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v6i.yolov11 \
  --config configs/yolov11_config.yaml \
  --murs-sains /content/drive/MyDrive/murs_sains

Reprendre un entraînement interrompu :

!python scripts/train_yolov11.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v6i.yolov11 \
  --resume outputs/yolov11/run/weights/last.pt

Surcharger des hyperparamètres à la volée :

!python scripts/train_yolov11.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v6i.yolov11 \
  --epochs 50 \
  --batch 16 \
  --imgsz 640 \
  --device 0

7 — Entraînement Mask R-CNN
!python scripts/train_maskrcnn.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v7i.coco-segmentation \
  --config configs/maskrcnn_config.yaml

Avec exemples négatifs :

!python scripts/train_maskrcnn.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v7i.coco-segmentation \
  --config configs/maskrcnn_config.yaml \
  --murs-sains /content/drive/MyDrive/murs_sains

Reprendre un entraînement interrompu (dernier checkpoint automatique) :

!python scripts/train_maskrcnn.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v7i.coco-segmentation \
  --resume

Reprendre depuis un checkpoint précis :

!python scripts/train_maskrcnn.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v7i.coco-segmentation \
  --resume-from outputs/maskrcnn/run/model_0005000.pth

Surcharger des hyperparamètres :

!python scripts/train_maskrcnn.py \
  --data-root /content/drive/MyDrive/segmentation_fissures.v7i.coco-segmentation \
  --max-iter 15000 \
  --batch 2 \
  --lr 0.00005 \
  --output-dir /content/drive/MyDrive/outputs/maskrcnn

8 — Évaluation
YOLOv11 :

!python scripts/evaluate.py \
  --model outputs/yolov11/run/weights/best.pt \
  --data-root /content/drive/MyDrive/segmentation_fissures.v6i.yolov11 \
  --split test

Mask R-CNN :

!python scripts/evaluate.py \
  --model outputs/maskrcnn/run/model_final.pth \
  --data-root /content/drive/MyDrive/segmentation_fissures.v7i.coco-segmentation \
  --split test

Métriques calculées : mAP@50, mAP@50-95, mAP@90, Précision, Rappel, F1, Mask mAP.

9 — Inférence
YOLOv11 sur une image :

!python scripts/inference.py \
  --model outputs/yolov11/run/weights/best.pt \
  --source /content/drive/MyDrive/images/mur.jpg

YOLOv11 sur un dossier :

!python scripts/inference.py \
  --model outputs/yolov11/run/weights/best.pt \
  --source /content/drive/MyDrive/images/ \
  --output-dir /content/drive/MyDrive/outputs/inference

Mask R-CNN :

!python scripts/inference.py \
  --model outputs/maskrcnn/run/model_final.pth \
  --source /content/drive/MyDrive/images/ \
  --num-classes 1

--num-classes doit correspondre à la valeur affichée par train_maskrcnn.py au démarrage (Classes réellement utilisées : [...]).

Désactiver l'analyse morphologique (inférence seule) :

!python scripts/inference.py \
  --model outputs/yolov11/run/weights/best.pt \
  --source /content/drive/MyDrive/images/ \
  --no-analysis

Sauvegarder les masques bruts en plus :

!python scripts/inference.py \
  --model outputs/yolov11/run/weights/best.pt \
  --source /content/drive/MyDrive/images/ \
  --save-masks

Ajuster la calibration mm/pixel :

!python scripts/inference.py \
  --model outputs/yolov11/run/weights/best.pt \
  --source /content/drive/MyDrive/images/ \
  --px-to-mm 0.3

10 — Analyse morphologique autonome
Sur un masque PNG existant :

!python scripts/crack_analysis.py \
  --mask /content/drive/MyDrive/masks/mur_mask.png \
  --image /content/drive/MyDrive/images/mur.jpg \
  --px-to-mm 0.5

Test sur masque synthétique (aucun argument requis) :

!python scripts/crack_analysis.py

11 — Copier les résultats vers Drive
import shutil
shutil.copytree('outputs', '/content/drive/MyDrive/outputs_fissures', dirs_exist_ok=True)

Datasets
Les datasets sont exportés depuis Roboflow :

Modèle	Format	Structure
YOLOv11	YOLOv11 (.txt)	train/images/ + train/labels/
Mask R-CNN	COCO JSON	train/_annotations.coco.json + train/
Note classes : Les exports Roboflow COCO incluent parfois une supercategory fantôme dans categories. Les scripts filtrent automatiquement pour ne conserver que les catégories réellement utilisées dans les annotations (NUM_CLASSES déduit du JSON, pas fixé en dur).

Note masques : cfg.INPUT.MASK_FORMAT = "bitmask" est imposé pour Mask R-CNN afin de tolérer les formats mixtes (polygones, RLE) produits par Roboflow.

Analyse morphologique
Le module crack_analysis.py calcule pour chaque instance :

Mesure	Méthode
Orientation	PCA → horizontale / verticale / inclinée
Localisation	Distance transform → superficielle / profonde / transversale
Longueur	Squelettisation (Zhang-Suen)
Largeur	Distance transform (médiane)
Sinuosité	Longueur squelette / distance terminaux
Sévérité	Classification normative par largeur (mm)
Indice danger	Score composite [0, 1]

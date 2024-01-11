# Classification de Chats et no_cats

Ce projet vise à réaliser une classification d'images de chats et non-chats. Les test on été effectué sur des datasets qui peuvent être téléchargé à partir des lien suivants : [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals?resource=download) et [Archive.org](https://archive.org/details/CAT_DATASET).

## Installation

1. Clonez ce dépôt sur votre machine locale.
2. Téléchargez le dataset Microsoft cats and no-cats à partir du lien mentionné ci-dessus.
3. Extrayez les fichiers du dataset dans le répertoire `data` du projet. 
4. Séparer les fichiers de la facon suivante :
```bash
> data
>> train
>>> no-cat
>>> cat
>> test
>>> no-cat
>>> cat
```

## Prérequis

Assurez-vous d'avoir les éléments suivants installés :

- Python 3.10
- Les bibliothèques nécessaires (voir le fichier `requirements.txt`)

## Utilisation

### Utilisation

Pour utiliser ce projet, vous pouvez exécuter le fichier `main.py` avec les arguments suivants :

- `--train` : pour entraîner le modèle.
   - `--epoch` : nombre d'époques d'entraînement (par défaut: 5).
   - `--batch_size` : taille du batch d'entraînement (par défaut: 32).
   - `--weight_name` : nom du fichier de sauvegarde des poids du modèle (par défaut: 'model').
   - `--learning_rate` : taux d'apprentissage du modèle (par défaut: 0.01).

- `--eval` : pour évaluer le modèle.
   - `--model_path` : chemin vers le fichier de poids du modèle (par défaut: 'weights/model.tf').
   - `--metrics` : liste des métriques à calculer (par défaut: ['accuracy', 'confusion_matrix', 'classification_report']).
   - `--no_save_cm` : désactiver la sauvegarde de la matrice de confusion (par défaut: True).
   - `--no_save_txt` : désactiver la sauvegarde du rapport de classification (par défaut: True).

- `--detect` : pour détecter une image.
   - `--image_path` : chemin vers l'image à détecter (obligatoire).
   - `--model_path` : chemin vers le fichier de poids du modèle (par défaut: 'weights/model.tf').

Si vous spécifiez plusieurs arguments parmi 'train', 'eval' et 'detect', une erreur sera levée.

Si vous ne spécifiez aucun arguments vosu pourrez acceder a l'interface pour effectuer une inference sur l'iamge et avec le poid de votre choix.

Exemples d'utilisation :
```bash

python main.py 
```
```bash

python main.py --eval --model_path weights/model_890.tf
```


## Information sur le modèle

1. **Couches de convolution (`Conv2D`) :**
   - Les couches de convolution sont responsables de l'extraction des caractéristiques des images. Elles utilisent des filtres pour détecter des motifs tels que les bords, les formes, et les textures.
   - Dans ce modèle, la première couche `Conv2D` a 32 filtres de taille (3, 3), ce qui signifie qu'elle applique 32 filtres 3x3 à l'image d'entrée.
   - La deuxième couche `Conv2D` a 64 filtres de taille (3, 3). Cela permet d'apprendre des caractéristiques plus complexes par rapport à la première couche.

2. **Couches de pooling (`MaxPooling2D`) :**
   - Les couches de pooling (mise en commun) réduisent la dimension spatiale des images et conservent les caractéristiques les plus importantes. Elles réduisent également le nombre de paramètres et le coût de calcul.
   - Dans ce modèle, après chaque couche de convolution, une couche `MaxPooling2D` avec une fenêtre de (2, 2) est utilisée. Cela réduit la taille de l'image de moitié à chaque fois.

3. **Couche de flattening (`Flatten`) :**
   - Cette couche transforme les données en un vecteur unidimensionnel. Elle prend la sortie de la dernière couche de pooling et la "déroule" pour l'aplatir avant de passer à des couches entièrement connectées.

4. **Couches entièrement connectées (`Dense`) :**
   - Ces couches utilisent tous les neurones pour apprendre des motifs complexes et effectuer la classification finale.
   - La première couche `Dense` a 128 neurones, activés par la fonction ReLU. Cela permet au modèle d'apprendre des combinaisons complexes de caractéristiques extraites par les couches précédentes.
   - La dernière couche `Dense` a 1 neurone avec une fonction d'activation `sigmoid` car il s'agit d'un problème de classification binaire (no_cat ou chat). La fonction `sigmoid` produit une sortie entre 0 et 1, indiquant la probabilité d'appartenance à la classe positive.




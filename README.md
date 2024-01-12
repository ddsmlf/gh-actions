# Classification de Chats et no_cats

Ce projet vise à réaliser une classification d'images de chats et non-chats. Les test on été effectué sur des datasets qui peuvent être téléchargé à partir des lien suivants : [Kaggle](https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals?resource=download) pour les chats et [Archive.org](https://archive.org/details/CAT_DATASET) pour les non-chats.

## Prérequis

Assurez-vous d'avoir les éléments suivants installés sur votre environnement :

- Python 3.10
- numpy==1.24.1
- Pillow==10.2.0
- scikit_learn==1.3.2
- tensorflow==2.15.0


## Installation

1. Clonez ce dépôt sur votre machine locale.
2. Téléchargez le(s) dataset(s) de votre choix.
3. Si les données sont déjà séparé ou que vous souahitez vous même le faire, extrayez les fichiers du dataset dans le répertoire `data` du projet. et séparer les fichiers de la facon suivante :
   ```
   > data
   >> train
   >>> no-cat
   >>> cat
   >> test
   >>> no-cat
   >>> cat
   ```
4. Sinon séparez les siplement en dossiers "cat" et "no-cat" dans un repertoir de votre choix, tel que "data_desoranizeds" par exemple.

## Utilisation

Pour utiliser ce projet, vous pouvez exécuter le fichier `run.py` avec les arguments suivants :

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

python run.py 
```
```bash

python run.py --eval --model_path weights/model_890.tf
```

### Entrainement

3. Si les données sont déjà séparé ou que vous souahitez vous même le faire, extrayez les fichiers du dataset dans le répertoire `data` du projet. et séparer les fichiers de la facon suivante :
```
> data
>> train
>>> no-cat
>>> cat
>> test
>>> no-cat
>>> cat
```

## Information sur le modèle

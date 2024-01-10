# Classification de Chats et Chiens

Ce projet vise à réaliser une classification d'images de chats et de chiens en utilisant le dataset Microsoft Cats and Dogs. Le dataset peut être téléchargé à partir du lien suivant : [Microsoft Cats and Dogs Dataset](https://www.microsoft.com/en-us/download/details.aspx?id=54765).

## Installation

1. Clonez ce dépôt sur votre machine locale.
2. Téléchargez le dataset Microsoft Cats and Dogs à partir du lien mentionné ci-dessus.
3. Extrayez les fichiers du dataset dans le répertoire `data` du projet. 
4. Séparer les fichiers de la facon suivante :
```
> data
>> train
>>> Dog
>>> Cat
>> test
>>> Dog
>>> Cat
```

## Prérequis

Assurez-vous d'avoir les éléments suivants installés :

- Python 3.10
- Les bibliothèques nécessaires (voir le fichier `requirements.txt`)

## Utilisation

1. Lancez le script `modele.py` en changant les lignes du if main == name pour entraîner ou l'évaluer le modèle de classification.
2. Utilisez le script `main.py` pour effectuer des prédictions sur de nouvelles images garec a l'interface.

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
   - La dernière couche `Dense` a 1 neurone avec une fonction d'activation `sigmoid` car il s'agit d'un problème de classification binaire (chien ou chat). La fonction `sigmoid` produit une sortie entre 0 et 1, indiquant la probabilité d'appartenance à la classe positive.


## Auteurs

- COLIN Mélissa
- Junior Bruce



"""

    IA rapide

"""

"""
-------------------------------------- Imports --------------------------------------
"""

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

"""
-------------------------------------- Variables --------------------------------------
"""
# Chemin vers le répertoire contenant les images de chiens et de chats
repertoire_chien = "data/Dog"
repertoire_chat = "data/Cat"

# Encodeur pour les labels
label_encoder = LabelEncoder()


"""
------------------------------------------ Fonction ---------------------------------------------
"""

# Fonction pour charger et prétraiter les images

def charger_images(repertoire, label):
    images = []
    labels = []
    for fichier in os.listdir(repertoire):
        chemin_fichier = os.path.join(repertoire, fichier)

        try:
            # Ajout d'une vérification du format de l'image avant de l'ouvrir
            if fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
                image = Image.open(chemin_fichier)
                image = image.convert('RGB')  # Convertit en mode RGB si l'image est en mode autre que RGB
                image = image.resize((64, 64))  # Redimensionne l'image
                image = img_to_array(image)
                images.append(image)
                labels.append(label)
            else:
                print(f"Le fichier {chemin_fichier} n'est pas dans un format d'image pris en charge.")
        except Exception as e:
            print(f"Erreur lors du traitement de l'image {chemin_fichier}: {str(e)}")

    return np.array(images), np.array(labels)

"""
------------------------------------------ Main ---------------------------------------------
"""

# Charger les images de chiens
print("Chargement des images de chiens...")
images_chien, labels_chien = charger_images(repertoire_chien, "chien")

# Charger les images de chats
print("Chargement des images de chats...")
images_chat, labels_chat = charger_images(repertoire_chat, "chat")

# Concaténer les données
print("Concaténer les données...")
X = np.concatenate((images_chien, images_chat), axis=0) # entrée
y = np.concatenate((labels_chien, labels_chat), axis=0) # sortie

# Diviser les données en ensembles d'entraînement et de test
print("Diviser les données en ensembles d'entraînement et de test...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Encoder les labels
print("Encoder les labels...")
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Normaliser les valeurs des pixels des images
print("Normaliser les valeurs des pixels des images...")
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0

# Créer le modèle
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Entraîner le modèle
model.fit(X_train_normalized, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

# Faire des prédictions sur l'ensemble de test
predictions = model.predict(X_test_normalized)
y_pred = (predictions > 0.5).astype(int)

# Évaluer la précision du modèle
accuracy = accuracy_score(y_test_encoded, y_pred)
print(f"Précision du modèle : {accuracy * 100:.2f}%")

import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class SimpleImageClassifier:
    def __init__(self):
        self.model = None
        self.label_encoder = LabelEncoder()

    def _charger_images(self, repertoire, label):
        images = []
        labels = []
        for fichier in os.listdir(repertoire):
            chemin_fichier = os.path.join(repertoire, fichier)

            try:
                if fichier.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = Image.open(chemin_fichier)
                    image = image.convert('RGB')
                    image = image.resize((64, 64))
                    image = img_to_array(image)
                    images.append(image)
                    labels.append(label)
                else:
                    print(f"Le fichier {chemin_fichier} n'est pas dans un format d'image pris en charge.")
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {chemin_fichier}: {str(e)}")

        return np.array(images), np.array(labels)

    def entrainer_modele(self, chemin_donnees):
        # Charger les images de chiens
        print("Chargement des images de chiens...")
        images_chien, labels_chien = self._charger_images(os.path.join(chemin_donnees, "Dog"), "chien")

        # Charger les images de chats
        print("Chargement des images de chats...")
        images_chat, labels_chat = self._charger_images(os.path.join(chemin_donnees, "Cat"), "chat")

        # Concaténer les données
        print("Concaténer les données...")
        X = np.concatenate((images_chien, images_chat), axis=0)
        y = np.concatenate((labels_chien, labels_chat), axis=0)

        # Diviser les données en ensembles d'entraînement et de test
        print("Diviser les données en ensembles d'entraînement et de test...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Encoder les labels
        print("Encoder les labels...")
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        # Normaliser les valeurs des pixels des images
        print("Normaliser les valeurs des pixels des images...")
        X_train_normalized = X_train / 255.0
        X_test_normalized = X_test / 255.0

        # Créer le modèle
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Compiler le modèle
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Entraîner le modèle
        print("Entraînement du modèle...")
        self.model.fit(X_train_normalized, y_train_encoded, epochs=10, batch_size=32, validation_split=0.2)

        # Sauvegarder les poids du modèle
        self.model.save_weights("modele_poids.h5")
        print("Les poids du modèle ont été sauvegardés.")

    def evaluer_modele(self, chemin_donnees_test, modele_poids):
        # Charger le modèle à partir des poids
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Charger les poids du modèle
        self.model.load_weights(modele_poids)

        # Charger les images de test
        # Charger les images de chiens
        print("Chargement des images de chiens...")
        images_chien, labels_chien = self._charger_images(os.path.join(chemin_donnees_test, "Dog"), "chien")

        # Charger les images de chats
        print("Chargement des images de chats...")
        images_chat, labels_chat = self._charger_images(os.path.join(chemin_donnees_test, "Cat"), "chat")

        # Concaténer les données
        print("Concaténer les données...")
        images_test = np.concatenate((images_chien, images_chat), axis=0)
        labels_test = np.concatenate((labels_chien, labels_chat), axis=0)

        # Encoder les labels de test
        print("Encoder les labels de test...")
        y_test_encoded = self.label_encoder.transform(labels_test)

        # Normaliser les valeurs des pixels des images de test
        print("Normaliser les valeurs des pixels des images de test...")
        X_test_normalized = images_test / 255.0

        # Faire des prédictions sur l'ensemble de test
        predictions = self.model.predict(X_test_normalized)
        y_pred = (predictions > 0.5).astype(int)

        # Évaluer la précision du modèle
        accuracy = accuracy_score(y_test_encoded, y_pred)
        print(f"Précision du modèle : {accuracy * 100:.2f}%")


    def inferer_image(self, image_path, modele_poids):
        # Charger le modèle à partir des poids
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Conv2D(64, (3, 3), activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))

        # Charger les poids du modèle
        self.model.load_weights(modele_poids)

        # Charger l'image et effectuer l'inférence
        image = Image.open(image_path)
        image = image.convert('RGB')
        image = image.resize((64, 64))
        image = img_to_array(image)
        image_normalized = image / 255.0
        image_input = np.expand_dims(image_normalized, axis=0)
        prediction = self.model.predict(image_input)
        print(prediction)
        label = "chien" if prediction[0, 0] > 0.5 else "chat"
        print(f"Prédiction : {label}")
        return label

# Exemple d'utilisation
if __name__ == "__main__":
    classifier = SimpleImageClassifier()
    #classifier.entrainer_modele("D:\\EPSI\\Bachelor\\B3\\int-cont\\gh-actions\\data\\train")
    classifier.evaluer_modele("D:\\EPSI\\Bachelor\\B3\\int-cont\\gh-actions\\data\\test", "modele_poids.h5")
    #classifier.inferer_image("D:\\EPSI\\Bachelor\\B3\\int-cont\\gh-actions\\data\\test\\Dog\\9152.jpg", "modele_poids.h5")
    
    #classifier.inferer_image("D:\\EPSI\\Bachelor\\B3\\int-cont\\gh-actions\\data\\test\\Cat\\9428.jpg", "modele_poids.h5")

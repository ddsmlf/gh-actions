from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from data_traitement import data_load

def entrainer_modele():
    X, y = data_load(type = "train")

    print("Créer le modèle...")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    print("Compiler le modèle...")
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("Entraînement du modèle...")
    model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

    print("Sauvegarder les poids du modèle...")
    model.save_weights("modele_poids.h5")
    print("Les poids du modèle ont été sauvegardés.")


if __name__ == "__main__":
    entrainer_modele()
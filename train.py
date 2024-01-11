from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from data_traitement import data_load
import os

def entrainer_modele(epoch=5, batch_size=32, weight_name="modele", learning_rate=0.01):
    weight_name = "weights/" + weight_name + ".h5"
    i=1
    while os.path.exists(weight_name):
        weight_name = weight_name[:-3] + "_"+ str(i) + ".h5"
        i += 1
    X, y = data_load(type="train")

    print("Créer le modèle...")
    model = Sequential()
    model.add(Flatten(input_shape=(64, 64, 3)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))

    # Utilisation de l'optimiseur SGD avec un taux d'apprentissage de 0.01 (vous pouvez ajuster si nécessaire)
    optimizer = SGD(learning_rate=learning_rate)

    print("Compiler le modèle...")
    # Utilisation de la perte catégorielle crossentropy et la métrique 'accuracy'
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    print("Entraînement du modèle...")
    model.fit(X, y, epochs=epoch, batch_size=batch_size, validation_split=0.2)

    print("Sauvegarder les poids du modèle...")
    model.save_weights(weight_name)
    print("Les poids du modèle ont été sauvegardés.")


if __name__ == "__main__":
    entrainer_modele(epoch=10)
    entrainer_modele(epoch=10, learning_rate=0.001, weight_name="modele_lr_0.001")
    entrainer_modele(epoch=10, learning_rate=0.1, weight_name="modele_lr_0.1")
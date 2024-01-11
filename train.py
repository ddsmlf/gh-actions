from tensorflow.keras.models import Sequential, save_model
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.optimizers import SGD
from utils.data_traitement import data_load
import os

def train_model(epoch=5, batch_size=32, weight_name="model", learning_rate=0.01):
    weight_name = "weights/" + weight_name
    i = 1
    while os.path.exists(weight_name):
        weight_name = weight_name[:-3] + "_" + str(i)
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

    print("Sauvegarder le modèle avec sa configuration...")
    model.save(weight_name + ".tf")
    print("Le modèle avec sa configuration a été sauvegardé.")


if __name__ == "__main__":
    train_model(epoch=10)

from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from data_traitement import data_load

def evaluer_modele(modele_poids):
    X, y = data_load(type = "test")
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.load_weights(modele_poids)

    predictions = model.predict(X)
    y_pred = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y, y_pred)
    print(f"Précision du modèle : {accuracy * 100:.2f}%")

if __name__ == "__main__":
    evaluer_modele("modele_poids.h5")
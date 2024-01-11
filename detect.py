from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

from data_traitement import data_load



def inferer_image(image_path, modele_poids):

    image = data_load(img_path = image_path)
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.load_weights(modele_poids)

    prediction = model.predict(image)
    print(prediction)
    label = "no-cat" if prediction[0, 0] > 0.5 else "chat"
    print(f"Prédiction : {label}")
    return label

if __name__ == "__main__":
    inferer_image("D:\\EPSI\\Bachelor\\B3\\int-cont\\gh-actions\\data\\test\\no-cat\\9152.jpg", "modele_poids.h5")
    inferer_image("D:\\EPSI\\Bachelor\\B3\\int-cont\\gh-actions\\data\\test\\cat\\9428.jpg", "modele_poids.h5")
import os

# Désactiver les options d'optimisation OneDNN
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tqdm import tqdm

def charger_images(repertoire, label):
    images = []
    labels = []
    for fichier in tqdm(os.listdir(repertoire), desc=f"Chargement des images pour le label '{label}'"):
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

def charger_image(path):
        image = Image.open(path)
        image = image.convert('RGB')
        image = image.resize((64, 64))
        image = img_to_array(image)
        image = image / 255.0
        image_input = np.expand_dims(image, axis=0)
        return image_input


def clear_terminal():
    """
    Efface le terminal, fonctionne sur Windows, Linux, et macOS.
    """
    os_name = os.name

    if os_name == 'nt':  # Windows
        os.system('cls')
    else:  # Linux and macOS
        os.system('clear')


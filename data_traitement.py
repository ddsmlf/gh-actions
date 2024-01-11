from utils.utils import *
import os
from sklearn.model_selection import train_test_split
import shutil
from PIL import Image
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical



_data_folder = "data"

def _is_image_file(file_path):
    try:
        with Image.open(file_path):
            return True
    except:
        return False

def _check_data_folder():
    """Vérifie si le dossier data est présent et s'il contient les fichiers nécessaires.

    Returns:
        Bool : True si le dossier data est présent et s'il contient les fichiers nécessaires, False sinon.
    """
    train_folder = os.path.join(_data_folder, "train")
    test_folder = os.path.join(_data_folder, "test")
    train_nocat_file = os.path.join(train_folder, "no-cat")
    train_cat_file = os.path.join(train_folder, "cat")
    test_nocat_file = os.path.join(test_folder, "no-cat")
    test_cat_file = os.path.join(test_folder, "cat")

    if os.path.exists(_data_folder) and os.path.isdir(_data_folder):
        if os.path.exists(train_folder) and os.path.isdir(train_folder):
            if os.path.exists(test_folder) and os.path.isdir(test_folder):
                if os.path.exists(train_nocat_file) and os.path.isdir(train_nocat_file) and len(os.listdir(train_nocat_file)) > 0:
                    if os.path.exists(train_cat_file) and os.path.isdir(train_cat_file) and len(os.listdir(train_cat_file)) > 0:
                        if os.path.exists(test_nocat_file) and os.path.isdir(test_nocat_file) and len(os.listdir(test_nocat_file)) > 0:
                            if os.path.exists(test_cat_file) and os.path.isdir(test_cat_file) and len(os.listdir(test_cat_file)) > 0:
                                return True

    return False

import os
import shutil

import os
import shutil

def _destructure_data(path):
    """Met toutes les données (image) qui sont les sous dossiers de path à la racine de path et supprime les sous dossiers

    Args:
        path (str): Chemin vers le dossier contenant les données organisées.
    
    Returns:
        Bool : True si les données ont été désorganisées.
    """
    
    def move_files(src, dest):
        for root, dirs, files in os.walk(src):
            for file in files:
                file_path = os.path.join(root, file)
                if _is_image_file(file_path):
                    shutil.move(file_path, os.path.join(dest, file))
                else:
                    os.remove(file_path)

    for file in os.listdir(path):
        file_path = os.path.join(path, file)

        if os.path.isdir(file_path):
            move_files(file_path, path)
            try:
                if not _is_image_file(file_path) :
                    shutil.rmtree(file_path)
            except OSError:
                # Le dossier n'a pas pu être supprimé, peut être non vide
                pass

    return True


def _organize_data(data_desorganized_path):
    """Organise les données dans le dossier data.

    Args:
        data_desorganized_path (str): Chemin vers le dossier contenant les données désorganisées.

    Returns:
        Bool : True si les données ont été organisées.
    """
    # tester si data_desorganized_path existe :

    if not os.path.exists(data_desorganized_path):
        raise ValueError(f"Le chemin d'accès {data_desorganized_path} est introuvable.")

    _data_folder = "data"
    train_folder = os.path.join(_data_folder, "train")
    test_folder = os.path.join(_data_folder, "test")
    train_nocat_file = os.path.join(train_folder, "no-cat")
    train_cat_file = os.path.join(train_folder, "cat")
    test_nocat_file = os.path.join(test_folder, "no-cat")
    test_cat_file = os.path.join(test_folder, "cat")

    if not os.path.exists(_data_folder):
        os.mkdir(_data_folder)
    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(test_folder):
        os.mkdir(test_folder)
    if not os.path.exists(train_nocat_file):
        os.mkdir(train_nocat_file)
    if not os.path.exists(train_cat_file):
        os.mkdir(train_cat_file)
    if not os.path.exists(test_nocat_file):
        os.mkdir(test_nocat_file)
    if not os.path.exists(test_cat_file):
        os.mkdir(test_cat_file)

    nocat_path = os.path.join(data_desorganized_path, "no-cat")
    cat_path = os.path.join(data_desorganized_path, "cat")

    _destructure_data(nocat_path)
    _destructure_data(cat_path)

    try :
        nocat_train, nocat_test = train_test_split(os.listdir(nocat_path), test_size=0.2, random_state=42)
        cat_train, cat_test = train_test_split(os.listdir(cat_path), test_size=0.2, random_state=42)
    except Exception as e :
        raise ValueError("Merci de séparer les données dans des dossiers cat et no-cat")

    for file in nocat_train:
        try :
            shutil.move(os.path.join(nocat_path, file), train_nocat_file)
        except shutil.Error as e :
            pass
    for file in cat_train:
        try :
            shutil.move(os.path.join(cat_path, file), train_cat_file)
        except shutil.Error as e :
            pass

    for file in nocat_test:
        try :
            shutil.move(os.path.join(nocat_path, file), test_nocat_file)
        except shutil.Error as e :
            pass
    for file in cat_test:
        try :
            shutil.move(os.path.join(cat_path, file), test_cat_file)
        except shutil.Error as e :
            pass

    return True


def _data_normalized(path_data):
    print("Chargement des images de no_cats...")
    images_no_cat, labels_no_cat = charger_images(os.path.join(path_data, "no-cat"), "no-cat")

    print("Chargement des images de chats...")
    images_chat, labels_chat = charger_images(os.path.join(path_data, "cat"), "chat")

    print("Concaténer les données...")
    X = np.concatenate((images_no_cat, images_chat), axis=0)
    Y = np.concatenate((labels_no_cat, labels_chat), axis=0)

    print("Encoder les labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(Y)

    # Assurez-vous que les étiquettes sont encodées correctement pour une classification binaire
    y_encoded = to_categorical(y_encoded, num_classes=2)

    print("Normaliser les valeurs des pixels des images...")
    X_normalized = X / 255.0

    return X_normalized, y_encoded



def data_load(type = None, img_path = None):
    if type != None:
        if not _check_data_folder() :
            data_desorganized_path = input("Les données ne sont pas organisées au bon format.\nChoisir un chemin ou vos données sont séparé en \"cat\" et \"no-cat\" : \n")
            _organize_data(data_desorganized_path)
        if type != "train" and type != "test":
            raise ValueError("Le type doit être train ou test")
        path_data = os.path.join(_data_folder, type)
        X, y = _data_normalized(path_data)
        return X, y
    else :
        if not os.path.exists(img_path):
            raise ValueError(f"Le chemin d'accès {img_path} est introuvable.")
        return charger_image(img_path)
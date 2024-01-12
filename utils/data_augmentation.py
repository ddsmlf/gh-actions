import cv2
import os
import subprocess
import matplotlib.pyplot as plt
from imgaug.augmenters import contrast as iaa_contrast
from imgaug import augmenters as iaa
import numpy as np
import random
from tqdm import tqdm

def rotation(image, angle):
    """
    Effectue une rotation de l'image selon l'angle spécifié.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        angle (float): L'angle de rotation en degrés.

    Returns:
        numpy.ndarray: L'image augmentée.
    """
    rotation_augmentor = iaa.Affine(rotate=angle)
    augmented_image = rotation_augmentor.augment_image(image)
    return augmented_image

def deformation(image, scale=(0.8, 1.2)):
    """
    Effectue une déformation de l'image avec une échelle aléatoire.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        scale (tuple): Plage d'échelle pour la déformation.

    Returns:
        numpy.ndarray: L'image augmentée.
    """
    deformation_augmentor = iaa.Affine(scale=scale)
    augmented_image = deformation_augmentor.augment_image(image)
    return augmented_image

def mosaic(image, size=(10, 10)):
    """
    Crée un effet de mosaïque sur l'image.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        size (tuple): La taille de la mosaïque, en pixels.

    Returns:
        numpy.ndarray: L'image augmentée avec l'effet de mosaïque.
    """
    mosaic_augmentor = iaa.CoarseDropout(0.2, size_percent=0.2)
    augmented_image = mosaic_augmentor.augment_image(image)
    return augmented_image

def obfuscation(image, strength=0.2):
    """
    Obscurcit l'image pour ajouter un effet d'obfuscation.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        strength (float): La force de l'effet d'obfuscation.

    Returns:
        numpy.ndarray: L'image augmentée avec l'effet d'obfuscation.
    """
    obfuscation_augmentor = iaa.Multiply((0.7, 1.3), per_channel=strength)
    augmented_image = obfuscation_augmentor.augment_image(image)
    return augmented_image

def superposition(image, overlay_image):
    """
    Superpose une deuxième image sur l'image principale.

    Parameters:
        image (numpy.ndarray): L'image principale.
        overlay_image (numpy.ndarray): L'image à superposer.

    Returns:
        numpy.ndarray: L'image résultante de la superposition.
    """
    superposition_augmentor = iaa.BlendAlpha(0.7, iaa.AllChannelsHistogramEqualization())
    overlay_augmented = superposition_augmentor.augment_image(overlay_image)
    augmented_image = cv2.addWeighted(image, 0.7, overlay_augmented, 0.3, 0)
    return augmented_image

def filtre_couleurs_lumieres(image, hue=(-20, 20), contrast=(0.5, 1.5), brightness=(0.5, 1.5)):
    """
    Ajoute des filtres de couleurs et ajuste la luminosité de l'image.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        hue (tuple): Plage de variation de la teinte.
        contrast (tuple): Plage de variation du contraste.
        brightness (tuple): Plage de variation de la luminosité.

    Returns:
        numpy.ndarray: L'image augmentée avec des filtres de couleurs et un ajustement de luminosité.
    """
    filtre_augmentor = iaa.Sequential([
        iaa.AddToHueAndSaturation(hue),
        iaa_contrast.LinearContrast(contrast),
        iaa.MultiplyBrightness(brightness)
    ])
    augmented_image = filtre_augmentor.augment_image(image)
    return augmented_image

def ajout_flou(image, sigma=(0.5, 1.5)):
    """
    Ajoute un flou gaussien à l'image.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        sigma (tuple): Plage de variation de l'écart-type du noyau du filtre gaussien.

    Returns:
        numpy.ndarray: L'image augmentée avec l'effet de flou gaussien.
    """
    blur_augmentor = iaa.GaussianBlur(sigma=sigma)
    augmented_image = blur_augmentor.augment_image(image)
    return augmented_image

def transformation_affine(image, scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), rotate=(-45, 45)):
    """
    Applique une transformation affine à l'image.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        scale (tuple): Facteur d'échelle de la transformation.
        translate_percent (tuple): Pourcentage de translation par rapport aux dimensions de l'image.
        rotate (tuple): Angle de rotation de la transformation.

    Returns:
        numpy.ndarray: L'image augmentée avec la transformation affine.
    """
    affine_augmentor = iaa.Affine(scale=scale, translate_percent=translate_percent, rotate=rotate)
    augmented_image = affine_augmentor.augment_image(image)
    return augmented_image

def inversion_couleurs(image):
    """
    Inverse les couleurs de l'image.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.

    Returns:
        numpy.ndarray: L'image augmentée avec l'inversion des couleurs.
    """
    inversion_augmentor = iaa.Invert(1.0, per_channel=True)
    augmented_image = inversion_augmentor.augment_image(image)
    return augmented_image

def ajout_bruit(image, scale=(0, 0.1)):
    """
    Ajoute du bruit gaussien à l'image.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        scale (tuple): Échelle du bruit gaussien.

    Returns:
        numpy.ndarray: L'image augmentée avec l'ajout de bruit gaussien.
    """
    # Utiliser une seule valeur pour scale au lieu d'une plage
    bruit_augmentor = iaa.AdditiveGaussianNoise(scale=scale[1]*255)
    augmented_image = bruit_augmentor.augment_image(image)
    return augmented_image


def rotation_aleatoire(image):
    """
    Effectue une rotation aléatoire de l'image.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.

    Returns:
        numpy.ndarray: L'image augmentée avec une rotation aléatoire.
    """
    rotation_augmentor = iaa.Affine(rotate=(-180, 180))
    augmented_image = rotation_augmentor.augment_image(image)
    return augmented_image

def ajuster_luminosite(image, gamma=(0.5, 1.5)):
    """
    Ajuste la luminosité de l'image en utilisant la correction gamma.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        gamma (tuple): Plage de variation de la valeur pour l'ajustement de la luminosité.

    Returns:
        numpy.ndarray: L'image augmentée avec l'ajustement de la luminosité.
    """
    luminosite_augmentor = iaa.GammaContrast(gamma=gamma)
    augmented_image = luminosite_augmentor.augment_image(image)
    return augmented_image

def decalage_couleur(image, rgb_shift=(-30, 30)):
    """
    Effectue un décalage des couleurs RGB de l'image.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        rgb_shift (tuple): Plage de variation du décalage des valeurs RGB.

    Returns:
        numpy.ndarray: L'image augmentée avec le décalage des couleurs.
    """
    decalage_augmentor = iaa.AddToHue(value=rgb_shift[0])
    augmented_image = decalage_augmentor.augment_image(image)
    return augmented_image



def distorsion_elastique(image, alpha=(0, 40), sigma=5):
    """
    Applique une distorsion élastique à l'image.

    Parameters:
        image (numpy.ndarray): L'image à augmenter.
        alpha (tuple): Amplitude de la distorsion élastique.
        sigma (int): Écart-type du noyau gaussien pour la distorsion élastique.

    Returns:
        numpy.ndarray: L'image augmentée avec la distorsion élastique.
    """
    distorsion_augmentor = iaa.ElasticTransformation(alpha=alpha, sigma=sigma)
    augmented_image = distorsion_augmentor.augment_image(image)
    return augmented_image


def _zoom_aleatoire(image, zoom_factor=(1.0, 1.5)):
    zoom_augmentor = iaa.Affine(scale=zoom_factor)
    augmented_image = zoom_augmentor.augment_image(image)
    return augmented_image

def _decoupage_aleatoire(image, percent=(0.1, 0.2)):
    decoupage_augmentor = iaa.Crop(percent=percent)
    augmented_image = decoupage_augmentor.augment_image(image)
    return augmented_image

def _save(image_augmentee, chemin_sortie, nom_fichier_original):
    """
    Sauvegarde l'image augmentée avec un nom de fichier modifié.

    Parameters:
        image_augmentee (numpy.ndarray): L'image augmentée à sauvegarder.
        chemin_sortie (str): Le chemin du dossier de sortie.
        nom_fichier_original (str): Le nom du fichier original.

    Returns:
        None
    """
    nom_fichier_parts = nom_fichier_original.split('.')
    
    if len(nom_fichier_parts) > 1:
        nom_base = '.'.join(nom_fichier_parts[:-1])
        extension = nom_fichier_parts[-1]
        nom_fichier_augmente = f"{nom_base}_augmentee.{extension}"
    else:
        nom_fichier_augmente = f"{nom_fichier_original}_augmentee"

    chemin_fichier_augmente = os.path.join(chemin_sortie, nom_fichier_augmente)
    cv2.imwrite(chemin_fichier_augmente, image_augmentee)


def _apply(image):
    fonctions_augmentations = [
        (rotation, (random.uniform(-180, 180),)),
        (deformation, ()),
        (mosaic, ()),
        (obfuscation, (random.uniform(0, 0.4),)),
        (superposition, (image,)),
        (filtre_couleurs_lumieres, (random.uniform(-20, 20), random.uniform(0.5, 1.5), random.uniform(0.5, 1.5))),
        (ajout_flou, (random.uniform(0.5, 1.5),)),
        (transformation_affine, ()),
        (inversion_couleurs, ()),
        (ajout_bruit, ((0, random.uniform(0, 0.1)),)),
        (rotation_aleatoire, ()),
        (ajuster_luminosite, (random.uniform(0.5, 1.5),)),
        (decalage_couleur, ((random.randint(-30, 30), random.randint(-30, 30), random.randint(-30, 30)),)),
        (distorsion_elastique, (random.uniform(0, 40),)),
        (_zoom_aleatoire, ()),
        (_decoupage_aleatoire, ()),
    ]

    # appliquer une modification aleatoire sur l'image :
    index = np.random.randint(0, len(fonctions_augmentations))
    fonction, args = fonctions_augmentations[index]
    image_augmentee = fonction(image, *args)

    return image_augmentee

def debug(image, images_path):
    try :
        image_augmentee = _apply(image)
    except :
        new_image = cv2.imread(images_path + "/" + random.choice(os.listdir(images_path)))
        image_augmentee = _apply(new_image)
    return image_augmentee
    
def _aug(images_path, quantity):
    for i in tqdm(range(quantity), desc="Augmenting images"):
        image_path = random.choice(os.listdir(images_path))
        image_path = os.path.join(images_path, image_path)
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)
        try :
            image_augmentee = _apply(image)
        except :
            new_image = cv2.imread(images_path + "/" + random.choice(os.listdir(images_path)))
            image_augmentee = debug(new_image, images_path)
        _save(image_augmentee, images_path, image_name)


def data_augmentation(path, quantity):
    quantity = int(quantity/2) 
    cat_path = os.path.join(path, "cat")
    _aug(cat_path, quantity)
    nocat_path = os.path.join(path, "no-cat")
    _aug(nocat_path, quantity)



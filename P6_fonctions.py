# **********************************************************************************************************************
# Liste des fonctions de ce module :
#
# - elapsed_format: affiche le temps écoulé
#
# - Dictionnaire anglais: 4 listes des mots anglais (miniscules, ponctuation)
# - Dictionnaire français: 4 listes des mots français (miniscules, ponctuation)
#
# - showImg: affiche une image avec OpenCV
# - resizeImg: redimensionne une image avec openCV
#
# - kmeans_metric_plot: trace les métriques dévaluation du clustering avec k-Means
#
# - vgg16: définition du CNN VGG16 avec keras
# **********************************************************************************************************************
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Display options
from IPython.display import display, display_html, display_png, display_svg

pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 199)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

# Colorama
from colorama import Fore, Back, Style
# Fore: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Back: BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE, RESET.
# Style: DIM, NORMAL, BRIGHT, RESET_ALL

# Répertoires de sauvegarde
data_path = './P6_data/Flipkart/'
fig_path = './P6_fig/'

# Création d'une cmap discrète jusque 10 couleurs  pour affichage de clusters
import matplotlib.colors as mcolors
discrete_palette = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
                    'tab:brown', 'tab:pink', 'tab:grey', 'tab:olive', 'tab:cyan']
def discrete_colormap(colors=['steelblue', 'coral']):
    cmap = mpl.colors.ListedColormap(colors).with_extremes(over=colors[0], under=colors[1])
    bounds = np.linspace(0, 1, len(colors))
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    return cmap, norm

#***********************************************************************************************************************

# Formate le temps écoulé entre 2 timeit.default_timer()
import timeit
import datetime
def elapsed_format(elapsed):
    duration = datetime.datetime.utcfromtimestamp(elapsed)
    if elapsed >= 3600:
        return f"{duration.strftime('%H:%M:%S')}"
    elif elapsed >= 60:
        return f"{duration.strftime('%M:%S')}"
    elif elapsed >=1:
        return f"{duration.strftime('%S.%f')[:-3]}s"
    else:
        return f"{duration.strftime('%f')[:-3]}ms"

#***********************************************************************************************************************
# Dictionnaire des mots anglais
# url='https://github.com/mwiens91/english-words-py',
# author='Matt Wiens',
# author_email='mwiens91@gmail.com',
# license='MIT',
# Development Status :: 5 - Production/Stable
# License :: OSI Approved :: MIT License

import os
# set of English words containing both upper- and lower-case letters; with punctuation
english_words_set = open(os.path.join(".", "english_words_dic.txt"), "r").read().split("\n")

# Set of English words containing lower-case letters; with punctuation
english_words_lower_set = list(w.lower() for w in english_words_set)

# Set of English words containing both upper- and lower-case letters; with no punctuation
english_words_alpha_set = set(english_words_set)

# Set of English words containing lower-case letters; with no punctuation
english_words_lower_alpha_set = set(english_words_lower_set)

words_dict = {"english_words_set": english_words_set,
              "english_words_lower_set": english_words_lower_set,
              "english_words_alpha_set": english_words_alpha_set,
              "english_words_lower_alpha_set": english_words_lower_alpha_set}

#***********************************************************************************************************************
# Dictionnaire des mots français
# Liste provenant de : https://www.freelang.com/download/misc/liste_francais.zip (mots accentués)
# license='MIT'

import os
# set of French words containing both upper- and lower-case letters; with punctuation
french_words_set = open(os.path.join(".", "french_words_dic.txt"), "r").read().split("\n")

# Set of French words containing lower-case letters; with punctuation
french_words_lower_set = list(w.lower() for w in french_words_set)

# Set of French words containing both upper- and lower-case letters; with no punctuation
french_words_alpha_set = set(french_words_set)

# Set of French words containing lower-case letters; with no punctuation
french_words_lower_alpha_set = set(french_words_lower_set)

french_words_dict = {"french_words_set": french_words_set,
                     "french_words_lower_set": french_words_lower_set,
                     "french_words_alpha_set": french_words_alpha_set,
                     "french_words_lower_alpha_set": french_words_lower_alpha_set}

#***********************************************************************************************************************
# Affichage d'une image avec opencv
import cv2 as cv

# Fonction d'affichage d'une image: name (str) et img la matrice image
def showImg(name, img):
    cv.namedWindow(name, cv.WINDOW_GUI_EXPANDED and  cv.WINDOW_KEEPRATIO)
    cv.imshow(name, img)
    cv.waitKey()

# Redimensionnement des images en 224 x 224
image_path = './P6_data/Flipkart/Images/'
resized_img_path = image_path + "Resized/"
def resizeImg(filename_list, target_size=(224, 224)):
    """
    Redimensionne les images de la liste à la taille cible
    et les place dans le répertoire resized_img_path.
    :param filename_list: str, liste des noms de fichier image
    :param target_size: 2-tuple, taille cible des images
    :return: None
    """
    for index, iname in enumerate(tqdm(filename_list, desc="Image resizing")):
        filename = image_path + iname
        img = cv.imread(filename)
        img = cv.resize(img, target_size, interpolation= cv.INTER_AREA)
        filename = resized_img_path + iname
        cv.imwrite(filename, img)

#***********************************************************************************************************************
#
# Détermination du nombre de clusters optimal avec K-Means
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics
import timeit
from tqdm.notebook import tqdm
eval_metrics = ['silhouette', 'calinski_harabasz', 'davies_bouldin']
def kmeans_metric_plot(X, ks=np.arange(2, 10), eval=eval_metrics, save=None):

    inertia = []
    if 'silhouette' in eval: silhouette = []
    if 'calinski_harabasz' in eval: calinski_harabasz = []
    if 'davies_bouldin' in eval: davies_bouldin = []

    for k in tqdm(ks):
        if len(X) < 10000:
            model = KMeans(n_clusters=k).fit(X)
        else:
            model = MiniBatchKMeans(n_clusters=k).fit(X)

        inertia.append(model.inertia_)
        if 'silhouette' in eval:
            silhouette.append(metrics.silhouette_score(X, model.labels_, metric='euclidean'))
        if 'calinski_harabasz' in eval:
            calinski_harabasz.append(metrics.calinski_harabasz_score(X, model.labels_))
        if 'davies_bouldin' in eval:
            davies_bouldin.append(metrics.davies_bouldin_score(X, model.labels_))

    # Affichage graphique du score en fonction du nombre de clusters
    nrows = int((len(eval)+1) / 2) + (len(eval)+1) % 2
    ncols = 2 if len(eval)>0 else 1
    plt.figure(figsize=(6*ncols, 4*nrows))

    plt.subplot(nrows, ncols, 1)
    plt.plot(ks, inertia)
    plt.xlabel('Nombre de clusters')
    plt.title("Inertie")

    i = 2
    if 'silhouette' in eval:
        plt.subplot(nrows, ncols, i)
        plt.plot(ks, silhouette)
        plt.xlabel('Nombre de clusters')
        plt.title("Silhouette")
        i += 1

    if 'calinski_harabasz' in eval:
        plt.subplot(nrows, ncols, i)
        plt.plot(ks, calinski_harabasz)
        plt.xlabel('Nombre de clusters')
        plt.title("Calinski-Harabasz")
        i += 1

    if 'davies_bouldin' in eval:
        plt.subplot(nrows, ncols, i)
        plt.plot(ks, davies_bouldin)
        plt.xlabel('Nombre de clusters')
        plt.title("Davies-Bouldin")

    plt.tight_layout()
    if save is not None:
        filename = fig_path + save + ".png"
        plt.savefig(filename, dpi=300)
    plt.show()

#***********************************************************************************************************************

#***********************************************************************************************************************
# Création du CNN architecture VGG-16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

def vgg16(X, Y, batch_size=256, epochs=1000):

    # Création du modèle
    fk_VGG16 = Sequential()

    # 1ère couche de convolution avec activation ReLU
    fk_VGG16.add(Conv2D(64, (3, 3), input_shape=(224, 224, 3), padding='same', activation='relu'))
    fk_VGG16.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    # 1ère couche de pooling
    fk_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 2ème couche de convolution, activation ReLU
    fk_VGG16.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    fk_VGG16.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    # 2ème couche de pooling
    fk_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 3ème couche de convolution, activation ReLU
    fk_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    fk_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    fk_VGG16.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    # 3ème couche de pooling
    fk_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 4ème couche de convolution, activation ReLU
    fk_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    fk_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    fk_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # 4ème couche de pooling
    fk_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # 5ème couche de convolution, activation ReLU
    fk_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    fk_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    fk_VGG16.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    # 5ème couche de pooling
    fk_VGG16.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conversion 3D → 1D
    fk_VGG16.add(Flatten())

    # 2 couches fully-connected
    fk_VGG16.add(Dense(4096, activation='relu'))
    fk_VGG16.add(Dense(4096, activation='relu'))

    # Dernière couche fully connected
    n_clusters = len(set(Y))
    fk_VGG16.add(Dense(n_clusters, activation='softmax'))

    # Compilation et entrainement
    fk_VGG16.compile(loss='categorical_crossentropy', optimizer='adam')
    fk_VGG16.fit(x=X, y=Y, batch_size=batch_size, epochs=epochs)

    return fk_VGG16

#***********************************************************************************************************************



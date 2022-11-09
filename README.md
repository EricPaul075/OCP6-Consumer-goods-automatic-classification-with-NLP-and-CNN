# OCP6-Consumer-goods-automatic-classification-with-NLP-and-CNN
Unsupervised classification of products based on their text description (NLP) or image (computer vision)

L'objet de ce projet est d'examiner la faisabilité de classifier de manière non supervisée la liste des produits d'un site de e-commerce à partir de leur description sous forme de texte ou d'image.

Le clustering s'effectue généralement selon le processus:
1- Contruction d'un jeu de données exploitable à partir du descriptif des produits ;
2- Réduction de dimension par projection dans le plan avec t-SNE ;
3- Clustering avec k-Means ;
4- Alignement des labels prédits et labels vrais avec la matrice de confusion ;
5- Représentation graphique comparée du clustering et des labels vrais (Matplotlib).

La construction d'un jeu de données en objet de la première étape de ce processus s'effectue alternativement à partir du descriptif textuel (NLP) des produits ou de leur image (Computer vision / CNN).

Le jeu de données construit par traitement du langage naturel (NLP) consiste tout d'abord à prétraiter le texte avec une batterie de traitements optionnels et paramétrables: transformation unicode et minuscule, découpage en phrase et mots, filtrages de ponctuation / stop-words / chiffres / mots courts / mots hors dictionnaire, et lemming et stemming.
A partir de ce prétraitement , les features sont établies sous la forme de type bag-of-words (comptage et fréquence des mots).
Alternativement aux étapes 2 et 3 du processus, le clustering du texte peut être effectuée avec le modèle de factorisation de matrice non négative (NMF) ou celui d'allocation de Dirichlet latente (LDA).

Outre cette approche basique mais très performante pour ce type de projet, d'autres approches NLP sont explorées:
- Word2Vec: association d'une fenêtre de contexte de part et d'autre à chaque mot, établissement du modèle (perceptron) de correspondance entre les mots et leur contexte, et vectorisation du descriptif de chaque produit avec ce modèle. Ce modèle ne s'est pas avéré pertinent pour ce projet.
- BERT (Bidirectional Encoder Representations from Transformers): réseau de neurone récursif (RNN), tenant compte du contexte des mots avec le mécanisme d'attention pour effectuer des prédictions gérant l'ambiguïté de sens, et disposant d'une variété de modèles pré-entrainés. 
- USE: Universal Sentence Encoder basé sur BERT pour examiner la similarité des phrases.

S'agissant du jeu de données construit à partir des images, deux approches sont explorées :
- SIFT (Scale-Invariant Feature Transform): extraction de bag of visual words à partir des points d'intérêt des images (coins et bords) ;
- CNN (Convolutional Neural Network): extraction de feature à partir de modèles CNN pré-entrainés (transfer learning).


Enfin, les approches NLP et CNN sont combinées de différentes manières pour consolider voire améliorer le résultat de la classification.
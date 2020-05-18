# Projet S7 Segmentation d’image réelle
#### Compte rendu de la journée du 20/11/2019 

Personnes présentes : Toute l’équipe

Suite à la première journée du 18/11, le travail à court-terme a déjà été réparti, ainsi chaque membre du groupe a travaillé en autonomie sur l’implémentation d’un algorithme différent pour la segmentation d’image.


## Segmentation par croissance de région :
Une première version de l’algorithme est implanté et fonctionnelle avec un prédicat assez simple et en ayant à fixer le différent points de départ, cependant il  reste à fixer la valeur de seuil à partir de laquelle on décide qu’un pixel appartient à la même région qu’un autre. Cela pourra se faire par apprentissage, ou en le fixant si les résultats sont convaincants. Il reste à affiner le prédicat en testant des prédicats locaux, ou en ayant un traitement différent des trois composantes RGB, et à poser les différentes seeds automatiquement.

## Segmentation par l’algorithme des lignes d’eau:
Une version de l’algorithme a été achevée en aide avec un article. Cet méthode nécessite de définir plusieurs paramètres: une tolérance sur l’intensité des pixels appartenant à un même bassin. Le choix d’un floutage préalable de l’image (application d’un filtre par une matrice contenant que des 1). Et enfin le réglage du plus petit bassin acceptable. Des tests ont été effectué pour plusieurs images, et les résultats sont satisfaisant pour la plupart. Cependant des réglages différents des paramètres cités précédemment sont parfois nécessaires. Les images qui posent problème sont celle contenant une multitude de petits détails, tandis que les images “simples” ( un objet avec un fond) sont reconnues sans difficultés.

## Segmentation par identification des bords :
5 premiers algorithmes ont été implémentés pour détecter les discontinuités sur une image : 3 opérateurs de premier ordre (de dérivation) et l'algorithme de Marr Hildreth (amélioration de l’opérateur Laplacien du Gaussien) et le détecteur Canny Edge. Le dernier donnant les résultats les plus satisfaisant dans de nombreux cas comme le disait la littérature. Il reste à implémenter un algorithme de segmentation par croissance des régions pour obtenir la segmentation en régions plutôt qu’en frontières. Le problème est que les bords obtenus ne sont pas souvent continus, il faut alors adapter un algorithme à la situation. Des travaux ont été commencés sur l’algorithme Split and Merge.

## Segmentation par graphe :
Une première version de l’algorithme a été implémentée selon l’article SegmentationParGraphe.pdf. Quelques tests ont déjà été réalisés et l’algorithme fonctionne mais différents paramètres restent à affiner pour améliorer les résultats. A noter que l’algorithme est relativement lent, notamment à cause de la phase de pré-traitement de l’image qui consiste à trouver pour chaque pixels lesquels de ses voisins sont relativement similaires. De plus l’algorithme a tendance à créer beaucoup de clusters. De plus jusqu’ici nous avons utilisé comme poids pour les arêtes la distance euclidienne en dimension 5 (3 pour les couleurs et 2 pour les coordonnées) mais il semblerait que faire des clusters pour chaque couleur avec la norme absolue puis faire l’intersection des clusters donne de meilleurs résultats et une implémentation a été commencée.




## Réseaux de neurones :
La journée a été utilisée pour se familiariser avec CNN et ses différents types de couches (convolutional et pooling layers).
Différents réseaux pré-entraînés de base possibles :
- ResNet : le plus performant mais très volumineux
- VGG-16 : plus vieux, moins performant, mais moins de couches donc plus rapide à entraîner
- MobileNet : très peu de couche donc très rapide à entraîner et à utiliser

Plusieurs architectures de réseau possibles :
- FCN : CNN en remplaçant les fully connected layers par des convolutional layers, et utilisation de skip connections pour faciliter l’upsampling
- SegNet : architecture sous forme d’encodeur-décodeur symétrique (le décodeur utilise les indices correspondant aux pooling layers de l’encodeur), mais pas de skip connections
- UNet : SegNet avec des skip connections
- PSPNet (Pyramid Scene Parsing Network) : le feature map donné par le réseau de base est downsampled à différentes échelles, puis la convolution est appliquée à toutes ces échelles. Toutes les feature map sont ensuite upscaled et superposées pour donner le résultat final après une dernière convolutional layer



Une fois que tous les algorithmes auront été implantés, notre objectif immédiat sera de prendre un échantillon d’images sur lequel tester nos algorithmes pour essayer d’établir les points forts et faibles de chacun, et de raffiner notre quantification de la qualité de résultat (notamment lorsqu’on a des images déjà traitées pour comparer).

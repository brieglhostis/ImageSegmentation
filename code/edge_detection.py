import numpy as np
from scipy import ndimage
import imageio
import matplotlib.pyplot as plt
import sys

def get_img(image_name):
    """
    Cette fonction permet d'obtenir le tableau des pixels pour une image du dataset
    :param image_name: (str) nom de l'image (.jpg inclu)
    :return: pic : (array) tableau des pixels de l'image
    """
    pic = imageio.imread("../data/Pets/images/"+image_name)
    return pic

def get_black_white_image(image_name):
    """
    Cette fonction permet d'obtenir le tableau des pixels en nuances de gris pour une image du dataset
    :param image_name: (str) nom de l'image (.jpg inclu)
    :return: gray : (array) tableau des pixels de l'image en nuances de gris
    """
    pic = imageio.imread("../data/Pets/images/" + image_name)

    gray = lambda rgb: np.dot(rgb[..., :3], [0.299, 0.587, 0.114])
    gray = gray(pic)

    return gray

def plot_image_from_table(table, name):
    """
    Cette fonction permet d'afficher l'image depuis le tableau de ses pixels
    :param table: (array) tableau de pixels de l'image
    :param name: (str) nom de l'image à afficher
    :return: None
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(table, cmap=plt.get_cmap(name='gray'))
    plt.title(name)
    #plt.show()

## Edege detection filters and algorithms

# Robert's operator

def robert_operator(image_name):
    """
    Cette fonction applique l'opérateur de Robert sur l'image,
    cet opérateur calcule la dérivée entre chaque couple de pixels pour caractériser leurs dissimilarité.
    :param image_name: (str) nom de l'image à étudier
    :return: G : (array) tableau des pixels de l'image après l'application de l'opérateur
    """
    grey = get_black_white_image(image_name)

    Kx = np.array([[-1, 0], [0, 1]], np.float32)
    Ky = np.array([[0, -1], [1, 0]], np.float32)

    Ix = ndimage.filters.convolve(grey, Kx)
    Iy = ndimage.filters.convolve(grey, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255

    #plot_image_from_table(G, "Robert's operator")

    return G

# Prewitt's operator

def prewitt_operator(image_name):
    """
    Cette fonction applique l'opérateur de Prewitt sur l'image,
    cet opérateur calcule les dérivées horizontales et verticales autour
    de chaque pixel pour caractériser leur dissimilarité.
    :param image_name: (str) nom de l'image à étudier
    :return: G : (array) tableau des pixels de l'image après l'application de l'opérateur
    """
    grey = get_black_white_image(image_name)

    Kx = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], np.float32)
    Ky = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], np.float32)

    Ix = ndimage.filters.convolve(grey, Kx)
    Iy = ndimage.filters.convolve(grey, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255

    #plot_image_from_table(G, "Prewitt's operator")

    return G

# Sobel's operator

def sobel_operator(image_name):
    """
    Cette fonction applique l'opérateur de Sobel sur l'image,
    cet opérateur calcule le gradient en chaque pixel pour caractériser leur dissimilarité.
    :param image_name: (str) nom de l'image à étudier
    :return: G : (array) tableau des pixels de l'image après l'application de l'opérateur
    """
    grey = get_black_white_image(image_name)

    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(grey, Kx)
    Iy = ndimage.filters.convolve(grey, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255

    #plot_image_from_table(G, "Sobel's operator")

    return G

# Marr Hildreth Algorithm

def marr_hildreth_algo(image_name, sigma=1):
    """
    Cette fonction applique l'algorithme de Marr Hildreth sur l'image,
    cet algorithme est une amélioration du filtre LoG (Laplacien du Gaussien) et est composé de 3 étapes :
    - Appliquer un filtre Gaussien à l'image pour la flouter;
    - Calculer le Laplacien discret en chaque pixel;
    - Cherhcer les pixels où le Laplacien passe par zero (le laplacien étant dérivateur d'ordre 2, il passe par zero pour les maximums du gradient).
    :param image_name: (str) nom de l'image à étudier
    :param sigma: (float) valeur de l'écart-type du fitre Gaussien
    :return: G : (array) tableau des pixels de l'image après l'application de l'opérateur
    """
    size = int(2*(np.ceil(3*sigma))+1)
    grey = get_black_white_image(image_name)

    x, y = np.meshgrid(np.arange(-size/2+1, size/2+1),
                       np.arange(-size/2+1, size/2+1))

    normal = 1 / (2.0 * np.pi * sigma**2)

    kernel = ((x**2 + y**2 - (2.0*sigma**2)) / sigma**4) * \
        np.exp(-(x**2+y**2) / (2.0*sigma**2)) / normal  # Filtre LoG

    kern_size = kernel.shape[0]

    # application du filtre
    log = ndimage.filters.convolve(grey, kernel)

    zero_crossing = np.zeros_like(log)

    # zero crossing
    for i in range(log.shape[0]-(kern_size-1)):
        for j in range(log.shape[1]-(kern_size-1)):
            if log[i][j] == 0:
                if (log[i][j-1] < 0 and log[i][j+1] > 0) or (log[i][j-1] < 0 and log[i][j+1] < 0) or (log[i-1][j] < 0 and log[i+1][j] > 0) or (log[i-1][j] > 0 and log[i+1][j] < 0):
                    zero_crossing[i][j] = 255
            if log[i][j] < 0:
                if (log[i][j-1] > 0) or (log[i][j+1] > 0) or (log[i-1][j] > 0) or (log[i+1][j] > 0):
                    zero_crossing[i][j] = 255

    #plot_image_from_table(zero_crossing, "Marr Hildreth algorithm")

    return zero_crossing

# Canny Edge Detector

def canny_edge_detector(image_name, max_d=3):
    """
    Cette fonction appilque la méthode de détection de Canny Edge, celle-ci est composée de 5 étapes :
    - Appliquer un filtre Gaussien pour flouter l'image;
    - Appliquer les filtres de Soble pour obtenir le gradient en chaque pixel;
    - Supprimer les pixels qui ne  sont pas des maximums locaux (mettre la valeur à 0);
    - Classifier les maximums locaux en maximums forts et faibles selon la valeur de la norme du gradient, supprimer les maximums négligeables;
    - Transformer les maximums faibles en maximums forts si ils sont dans la continuité d'un bord fort, supprimer les maximums faibles restant;
    - (bonus) Lier les bords proches pour éviter la discontinuité des bords (et réappliquer l'algorithme pour affiner les bords).
    :param image_name: (str) nom de l'image à étudier
    :return: step_5_bis_pic: (array)
    """

    # Filtre gaussien

    def gaussian_kernel(size, sigma=1):
        """
        Cette fonction calcule la matrice de l'opérateur Gaussien
        :param size: (int) taille de la matrice
        :param sigma: (float) écart-type de l'opérateur Gaussien
        :return: g: (array) matrice de l'opérateur Gaussien
        """
        size = int(size) // 2
        x, y = np.mgrid[-size:size + 1, -size:size + 1]
        normal = 1 / (2.0 * np.pi * sigma ** 2)
        g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
        return g

    ## Filtre de Sobel

    def sobel_filters(img):
        """
        Cette fonction applique les filtres de Sobel (norme et angle du gradient)
        :param img: (array) tableau des pixels de l'image à traiter
        :return: G: (array) tableau de la norme du gradient à chaque pixel
        :return: theta: (array) tableau de l'angle du gradient à chaque pixel
        """
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.filters.convolve(img, Kx)
        Iy = ndimage.filters.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)

        return (G, theta)

    # Non max suppression

    def non_max_suppression(img, D):
        """
        Cette fonction supprime les pixels pour lesquels la norme du gradient est élevée
        mais qui ne sont pas des maximums locaux. Pour cela, on compare la norme du gradient de chaque pixel
        avec celle du pixel en direction de l'angle du gradient.
        :param img: (array) tableau des normes des gradients des pixels de l'image
        :param D: (array) tableau des angles (en radian) du gradient en chaque pixel
        :return: Z: (array) tableau de l'image traitée où les pixels non maximums locaux sont en noir,
                            les maximums locaux sont représentés par la norme de leurs gradients
        """
        M, N = img.shape
        Z = np.zeros((M, N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1, M - 1):
            for j in range(1, N - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img[i, j + 1]
                        r = img[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img[i + 1, j - 1]
                        r = img[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img[i + 1, j]
                        r = img[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img[i - 1, j - 1]
                        r = img[i + 1, j + 1]

                    if (img[i, j] >= q) and (img[i, j] >= r):
                        Z[i, j] = img[i, j]
                    else:
                        Z[i, j] = 0

                except IndexError as e:
                    pass

        return Z

    # Threshold

    def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
        """
        Cette fonction classifie les maximums locauxselon la valeur de leur gradient:
        les maximums forts : maximums avec la norme de gradient la plus élevée,
        les maximums faibles : maximums avec une norme de gradient plus faible,
        les maximums supprimés : maximums avec une norme de gradient négligeable
        :param img: (array) tableau de l'image traitée par la focntion précédente (non_max_suppression)
        :param lowThresholdRatio: (float) seuil minimum pour sélectionner les maximums faibles
        :param highThresholdRatio: (float) seuil minimum pour sélectionner les maximums forts
        :return: res: (array) tableau des pixels classifiers selon leurs valeurs : 255 pour les maximums forts, 25 pour les maximums faibles, 0 pour le reste
        :return: weak: (int) valeur des pixels de maximums faibles (25 de base)
        :return: strong: (int) valeur des pixels de maximums forts (255 de base)
        """
        highThreshold = img.max() * highThresholdRatio;
        lowThreshold = highThreshold * lowThresholdRatio;

        M, N = img.shape
        res = np.zeros((M, N), dtype=np.int32)

        weak = np.int32(25)
        strong = np.int32(255)

        strong_i, strong_j = np.where(img >= highThreshold)
        zeros_i, zeros_j = np.where(img < lowThreshold)

        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res, weak, strong)

    # hysteresis

    def hysteresis(img, weak, strong=255):
        """
        Cette fonction permet de transformer les maximums faibles en maximums forts si ils sont dans la continuité de maximums forts
        afin d'obtenir des bords plus long (pour un soucis de continuité), puis supprime les maximums faibles restants
        :param img: (array) tableau de l'image traitée par la fonction précédente (threshold)
        :param weak: (int) valeur des pixels de maximums faibles (25 de base)
        :param strong: (int) valeur des pixels de maximums forts (255 de base)
        :return: img: (array) tableau de l'image traitée où les bords sont en blanc (255) et le reste est en noir (0)
        """
        M, N = img.shape
        for i in range(1, M - 1):
            for j in range(1, N - 1):
                if (img[i, j] == weak):
                    try:
                        if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                                or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                                or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                        img[i - 1, j + 1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img

    # bonus : lien entre les pixels non liés mais proches

    def link_borders(img, max_d):
        """
        Cette fonction permet de lier les bords proches mais non reliés pour éviter les risques de discontinuité
        et améliorer la performance de l'identification de régions dans l'image.
        :param img: (array) tableau de l'image où les bords sont en blanc (255) et le reste est en noir (0)
        :return: img: (array) tableau de l'image traitée
        """

        def no_link(img, i1, j1, i2, j2):
            """
            Cette fonction vérifie qu'il n'y ait pas de lien (bord) entre deux pixels
            :param img: (array) tableau de l'image où les bords sont en blanc (255) et le reste est en noir (0)
            :param i1: (int) indice de la ligne du premier pixel
            :param j1: (int) indice de la colonne du premier pixel
            :param i2: (int) indice de la ligne du second pixel
            :param j2: (int) indice de la colonne du second pixel
            :return: (bool) False si les pixels sont reliés par un bord
            """
            shape = img.shape
            for i1_ in range(max(i1 - 1, 0), min(i1 + 1, shape[0])):
                for j1_ in range(max(j1 - 1, 0), min(j1 + 1, shape[1])):
                    if abs(i2 - i1_) < 2 and abs(j2 - j1_) < 2:
                        return False
            return True

        def search_neighbor(img, i, j, max_d):
            """
            Cette fonction cherche pour des pixels d'un bord proche du pixel sélectionné et les relie si ce n'est pas déjà le cas
            :param img: (array) tableau de l'image où les bords sont en blanc (255) et le reste est en noir (0)
            :param i: (int) indice de la ligne d'un pixel qui appartient à un bord
            :param j: (int) indice de la colonne de ce même indice
            :param max_d: (int) distance maximale entre deux pixels à relier (3 de base)
            :return: img: (array) tableau de l'image traitée où les bords sont en blanc (255) et le reste est en noir (0)
            """
            shape = img.shape
            for i_ in range(max(i - max_d, 0), min(i + max_d, shape[0])):
                for j_ in range(max(j - max_d, 0), min(j + max_d, shape[1])):
                    if abs(i - i_) > 1 and abs(j - j_) > 1 and no_link(img, i, j, i_, j_):
                        if img[i_][j_] == 255:
                            i__, j__ = int(i_), int(j_)
                            while abs(i - i__) > 0 and abs(j - j__) > 0:
                                i__ += int((i - i__) / abs(i - i__))
                                j__ += int((j - j__) / abs(j - j__))
                                img[i__][j__] = 255
                            if abs(i - i__) == 0:
                                while abs(j - j__) > 0:
                                    j__ += int((j - j__) / abs(j - j__))
                                    img[i__][j__] = 255
                            if abs(j - j__) == 0:
                                while abs(i - i__) > 0:
                                    i__ += int((i - i__) / abs(i - i__))
                                    img[i__][j__] = 255

            return img

        shape = img.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if img[i][j] == 255:
                    img = search_neighbor(img, i, j, max_d)
        return img

    grey = get_black_white_image(image_name)
    step_1_pic = ndimage.filters.convolve(grey, gaussian_kernel(5,1))
    step_2_pic, theta = sobel_filters(step_1_pic)
    step_3_pic = non_max_suppression(step_2_pic, theta)
    step_4_pic, weak, strong = threshold(step_3_pic)
    step_5_pic = hysteresis(step_4_pic, weak, strong)

    step_5_bis_pic = link_borders(step_5_pic, max_d)

    return step_5_bis_pic

## Region identification

# Unseeded Region Growing

def unseeded_region_growing(region):
    """
    Applique l'algorithme de Unseeded Region Growing qui creé des zones progressivement
    en sélectionnant des pixels non colorés et en étendant des zones autours d'eux jusqu'à rencontrer un bord
    :param region: (array) tableau des pixels avec les bords en blanc (255) et le reste en noir (0)
    :return: region: (array) tableau des pixels colorés selon leurs régions
    """

    def paint_neighbors(region, i, j, color):
        """
        Cette fonction peint les voisins du pixel (i,j) dans la même couleur que celui-ci si ils ne sont pas déjà peints
        et renvoie la liste des pixels peints
        :param region: (array) tableau des pixels de l'image avec leurs couleurs
        :param i: (int) indice de la ligne du pixel initial
        :param j: (int) indice de la colonne du pixel initial
        :param color: (int) valeur de la couleur à peindre
        :return: region: (array) nouveau tableau des pixels colorés
        :return: pixels: (list) liste des indices des pixels nouvellement colorés
        """
        pixels = []
        shape = region.shape
        if i > 0 and region[i - 1][j] < 255:
            if region[i - 1][j] != color:
                region[i - 1][j] = color
                pixels.append((i - 1, j))
        if i < shape[0] - 1 and region[i + 1][j] < 255:
            if region[i + 1][j] != color:
                region[i + 1][j] = color
                pixels.append((i + 1, j))
        if j > 0 and region[i][j - 1] < 255:
            if region[i][j - 1] != color:
                region[i][j - 1] = color
                pixels.append((i, j - 1))
        if j < shape[1] - 1 and region[i][j + 1] < 255:
            if region[i][j + 1] != color:
                region[i][j + 1] = color
                pixels.append((i, j + 1))

        return region, pixels

    def interative_spreading(region, i, j, color):
        """
        Cette fonction étend  récursivement la zone à peindre de proche en proche depuis le pixel (i,j)
        :param region: (array) tableau des pixels de l'image avec leurs couleurs
        :param i: (int) indice de la ligne du pixel initial
        :param j: (int) indice de la colonne du pixel initial
        :param color: (int) valeur de la couleur à peindre
        :return: region: (array) nouveau tableau des pixels colorés
        """
        region[i][j] = color
        region, pixels = paint_neighbors(region, i, j, color)
        while len(pixels) > 0:
            next_pixels = []
            for i_, j_ in pixels:
                region, next_pixels_ = paint_neighbors(region, i_, j_, color)
                next_pixels += next_pixels_
            pixels = next_pixels

        return region

    def urg_subfunction(region):
        """
        Cette fonction sélectionne un pixel non coloré et colore la zone autour de lui jusqu'à rencontrer un bord
        :param region: (array) tableau des pixels de l'image avec leurs couleurs
        :return: region: (array) nouveau tableau des pixels colorés
        """
        shape = region.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if region[i][j] < 255:
                    i_, j_ = i, j
                    break
        color = region.max() + 1
        region = interative_spreading(region, i_, j_, color)

        return region

    while region.min() < 255:
        region = urg_subfunction(region)

    #plot_image_from_table(region, "Region Growing")

    region = (region - region.min())#/(region.max()-region.min()) * 255

    return region

# Get regions

def get_regions(region):
    """
    Cette fonction retrouve les régions délimités par des bords
    :param region: (array) tableau des pixels avec les bords en blanc (255) et le reste en noir (0)
    :return: final_regions: (list) liste des régions identifiées, chaque région est une liste des pixels dans la région
    """
    shape = region.shape
    regions = [[] for i in range(1,region.max()+1)]
    for i in range(shape[0]):
        for j in range(shape[1]):
            color = region[i][j]
            if color != 0:
                regions[color-1].append((i,j))
    final_regions = []
    for region_ in regions :
        if len(region_)>0:
            final_regions.append(region_)
    return final_regions

# Segmentation

def clustering_segmentation(image_name, region):
    """
    Cette fonction propose une segmentation de l'image en régions distinctes,
    il utilise pour cela un algorithme de clustering sur les régions pour les regrouper selon la similarité de leur teinte
    :param image_name: (str) nom de l'image à étudier
    :param region: (array) tableau des pixels caractérisés par un entier (de 0 au nombre de régions -1) définissant la région à laquelle le pixel appartient
    :return: region: (array) nouveau tableau des pixels colorés selon leur région d'appartenance
    """

    def average_color(img, region_):
        """
        Cette fonction calcule la nuance de gris moyenne au sein d'une région de l'image de départ.
        :param img: (array) tableau des pixels de l'image, caractérisés par leur nuance de gris (des floats)
        :param region_: (list) liste de coordonnées de pixels définissant une région de l'image
        :return: (float) nuance de gris moyenne dans la région
        """
        return sum([img[x][y] for (x,y) in region_])/len(region_)

    def k_clustering_reals(colors, k):
        """
        Cette fonction réalise un clustering des float d'une liste en k clusters
        :param colors: (list) liste des floats à ranger dans les clusters
        :param k: nombre de clusters à identifier
        :return: clusters: (list) la liste des k clusters, chaque cluster étant défini par la liste des indices de ces éléments dans la liste "colors"
        """

        def closest_cluster(color, colors, centers):
            """
            Cette fonction identifie le cluster le plus proche pour un float en particulier
            :param color: (float) élément à raccorder à un cluster
            :param colors: (list) liste des floats à ranger dans les clusters
            :param centers: (list) liste des centres des clusters définis par leurs indices (int) dans colors
            :return: closest_center: (int) indice (dans colors) du centre de cluster le plus proche
            """
            min_dist = sys.maxsize
            closest_center = 0
            for center in centers:
                dist = abs(color - colors[center])
                if dist < min_dist :
                    min_dist = dist
                    closest_center = center
            return closest_center

        def new_clusters(colors, centers):
            """
            Cette fonction créé les clusters à partir de la liste des centres qui les définissent
            :param colors: (list) liste des float à ranger dans les clusters
            :param centers: (list) liste des centres des clusters définis par leurs indices (int) dans colors
            :return: clusters: (list) liste des clusters identifiés, définis par la liste des indices (dans colors) de leurs membres
            """
            clusters = [[] for x in centers]
            for i in range(len(colors)):
                closest_cluster_ = closest_cluster(colors[i], colors, centers)
                clusters[centers.index(closest_cluster_)].append(i)
            return clusters

        def new_centers(clusters, colors):
            """
            Cette fonction trouve les nouveaux centres (point le plu proche du centroid) des clusters
            :param clusters: (list) liste des clusters, définis par la liste des indices (dans colors) de leurs membres
            :param colors: (list) liste des float à ranger dans les clusters
            :return: new_centers: (list) liste des centres des clusters définis par leurs indices (int) dans colors
            """
            new_centers_ = []
            for cluster in clusters:
                dist_sum = [0 for x in cluster]
                for i in cluster:
                    for j in cluster:
                        if i!=j:
                            dist_sum[cluster.index(i)] += abs(colors[i]-colors[j])
                dist_moy = [sum_/len(cluster) for sum_ in dist_sum]

                center = cluster[dist_moy.index(min(dist_moy))]
                new_centers_.append(center)
            return new_centers_

        if k < len(colors):
            centers = [int((i/k)*len(colors)) for i in range(k)]
            clusters = new_clusters(colors, centers)
            new_centers_ = new_centers(clusters, colors)
            while centers != new_centers_:
                centers = new_centers_
                clusters = new_clusters(colors, centers)
                new_centers_ = new_centers(clusters, colors)
            return clusters
        else :
            return [[i] for i in range(len(colors))]

    def clustering_reals(colors, k_max = 8):
        """
        Cette fonction cherche le meilleur clustering des float d'une liste en essayant plusieurs valeurs de k
        :param colors: (list) liste des floats à ranger dans les clusters
        :param k_max: nombre maximal de clusters à identifier
        :return: best_clusters: (list) la liste des clusters, chaque cluster étant défini par la liste des indices de ces éléments dans la liste "colors"
        """

        def average_distance(clusters, colors):
            """
            (Optionel)
            Cette fonction calcule la plus petite distance entre les moyennes des clusters
            :param clusters: (list) la liste des clusters, chaque cluster étant défini par la liste des indices de ces éléments dans la liste "colors"
            :param colors: (list) liste des floats à ranger dans les clusters
            :return: (float) distance entre les moyennes des clusters
            """
            average_colors = []
            for cluster in clusters:
                average_colors.append(sum([colors[i] for i in cluster])/len(cluster))
            color_distances = []
            for i in range(len(average_colors)):
                color_distances.append(min([abs(average_colors[i]-average_colors[j]) for j in range(len(average_colors)) if i!=j]))
            return min(color_distances)

        def minimal_distance(clusters, colors):
            """
            Cette fonction calcule la distance minimale entre les clusters
            :param clusters: (list) la liste des clusters, chaque cluster étant défini par la liste des indices de ces éléments dans la liste "colors"
            :param colors: (list) liste des floats à ranger dans les clusters
            :return: (float) distance minimale entre les clusters
            """
            distances = []
            for cluster in clusters:
                for i in cluster:
                    distances.append(min([min([abs(colors[i]-colors[j]) for j in cluster_]) for cluster_ in clusters if cluster_ != cluster]))
            return min(distances)

        if k_max > 2:
            min_score = 0
            best_clusters = []
            for k in range(3,k_max+1):
                clusters = k_clustering_reals(colors, k)
                min_dist = minimal_distance(clusters, colors)
                print("Pour",k,"clusters, la distance entre les clusters les plus proches est :", min_dist)
                if min_dist > min_score:
                    min_score = min_dist
                    best_clusters = clusters
            return best_clusters
        else :
            return k_clustering_reals(colors, 3)

    def merge_clusters(regions, clusters):
        new_regions = [[] for cluster in clusters]
        for i in range(len(clusters)):
            for index in clusters[i]:
                new_regions[i] = new_regions[i] + regions[index]
        return new_regions

    def repaint(region, regions):
        for i in range(len(regions)):
            for (x,y) in regions[i]:
                region[x][y] = i
        return region

    img = get_black_white_image(image_name)
    regions = get_regions(region)
    print("nombre de régions :", len(regions))
    colors = [average_color(img, region_) for region_ in regions]
    clusters = clustering_reals(colors)
    print("nombre de clusters :", len(clusters))
    new_regions = merge_clusters(regions, clusters)
    region = repaint(region, new_regions)

    return region

def deux_segmentation(region):
    """
    Cette fonction propose une segmentation de l'image en 2 régions
    :param region: (array) tableau des pixels avec les bords en blanc (255) et le reste en noir (0)
    :return: region: (array) nouveau tableau des pixels colorés selon leur région d'appartenance
    """
    shape = region.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if region[i][j]>1:
                region[i][j] = 1
            else : region[i][j] = 0
    return region

## Full algorithms

def canny_edge_deux_segmentation(image_name):
    """
    Cette fonction effectue la détection de bords Canny Edge puis cherche une segmentation en deux zones: l'objet et le fond
    :param image_name: (str) nom du fichier de l'image en format jpg
    """
    region = canny_edge_detector(image_name, max_d=3)
    # plot_image_from_table(region,image_name)
    region = unseeded_region_growing(region)
    plot_image_from_table(region, image_name)
    region = deux_segmentation(image_name, region)
    plot_image_from_table(region, image_name)

    plt.show()

def canny_edge_clustering_segmentation(image_name):
    """
    Cette fonction effectue la détection de bords Canny Edge puis cherche une segmentation en appliquant un algorithme de clustering sur els régions délimités par les bords
    :param image_name: (str) nom du fichier de l'image en format jpg
    """
    region = canny_edge_detector(image_name, max_d=3)
    #plot_image_from_table(region, image_name)
    region = unseeded_region_growing(region)
    #plot_image_from_table(region, image_name)
    region = clustering_segmentation(image_name, region)
    plot_image_from_table(region, image_name)

    plt.show()

## Comparaison

def multiple_plots(image_name):
    """
    Cette fonction affiche les résultats de segmentation et de détection des bords pour les filtres et algorithmes suivants :
    Robert's Operator, Prewitt's Operator, Sobel's Operator, Marr Hildreth Algorithm et Canny Edge Detector
    :param image_name: (str) nom de l'image à étudier
    """

    print("Getting image ...")
    img = get_img(image_name)
    print("Applying Robert's Operator ...")
    rob = robert_operator(image_name)
    print("Applying Prewitt's Operator ...")
    pre = prewitt_operator(image_name)
    print("Applying Sobel's Operator ...")
    sob = sobel_operator(image_name)
    print("Applying Marr Hildreth Algorithm ...")
    mha = marr_hildreth_algo(image_name)
    print("Getting regions from Marr Hildreth Algorithm ...")
    reg_mha = unseeded_region_growing(mha.copy())
    print("Applying Canny Edge Detector ...")
    ced = canny_edge_detector(image_name)
    print("Getting regions from Canny Edge Detector ...")
    reg_ced = unseeded_region_growing(ced.copy())

    plots = [(img,"Initial Image"),
             (rob,"Robert's Operator"),
             (pre,"Prewitt's Operator"),
             (sob,"Sobel's Operator"),
             (mha,"Marr Hildreth Algorithm"),
             (reg_mha,"Region Growing from Marr Hildreth Algorithm"),
             (ced,"Canny Edge Detector"),
             (reg_ced,"Region Growing from Canny Edge Detector")]

    for i in range(len(plots)):
        table, name = plots[i]
        plt.subplot(4, 2, i+1)
        plt.imshow(table, cmap=plt.get_cmap(name='gray'))
        plt.tick_params(
            which='both',
            left=False,
            right=False,
            bottom=False,
            top=False,
            labelleft=False,
            labelbottom=False)
        plt.xlabel(name)

    print("Done")

    plt.show()

## Tests


image_name_1 = "Ragdoll_3.jpg"
image_name_2 = "Bombay_31.jpg"
image_name_3 = "basset_hound_60.jpg"

image_name_4 = "american_pit_bull_terrier_62.jpg"
image_name_5 = "Russian_Blue_123.jpg"
image_name_6 = "beagle_82.jpg"

image_name_7 = "japanese_chin_37.jpg"
image_name_8 = "Chihuahua_41.jpg"
image_name_9 = "Bengal_195.jpg"

"""
robert_operator(image_name)
prewitt_operator(image_name)
sobel_operator(image_name)
marr_hildreth_algo(image_name)
"""

#canny_edge_deux_segmentation(image_name_1)
canny_edge_clustering_segmentation(image_name_1)
#print(comparing_result_name_pets(region, image_name_1))


#multiple_plots("japanese_chin_37.jpg")
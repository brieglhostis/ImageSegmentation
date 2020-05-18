# -*- coding: latin-1 -*-
import numpy as np
from PIL import Image
import scipy
import scipy.ndimage
import scipy.optimize
import os
import bisect
from math import sqrt
from math import ceil
from math import log
from math import exp
from seaborn import heatmap
import matplotlib.pyplot as plt

class UnionFind :
    """
    Une simple implémentation d'une structure UnionFind avec en plus la taille de chaque structure
    """
    def __init__(self) :
        self.parent = self
        self.rank = 0
        self.maxWeight = 0
        self.size = 1
    
    def find(self) :
        if self.parent == self :
            return self
        else :
            self.parent = self.parent.find()
            return self.parent

    def union(self, y, maxWeight) :
        selfRoot = self.find()
        yRoot = y.find()
        if selfRoot.rank > yRoot.rank :
            yRoot.parent = selfRoot
            selfRoot.size += yRoot.size
            selfRoot.maxWeight = maxWeight

        elif selfRoot.rank < yRoot.rank :
            selfRoot.parent = yRoot
            yRoot.size += selfRoot.size
            yRoot.maxWeight = maxWeight

        elif selfRoot != yRoot :
            yRoot.parent = selfRoot
            selfRoot.rank += 1
            selfRoot.size += yRoot.size
            selfRoot.maxWeight = maxWeight

def import_image(image_name):
    """
    Importe une image du dataset pet à partir du nom du fichier en str, renvoie un array numpy de tableaux de taille 3 encodant le RGB de chaque pixel
    Note : dans une version ultérieure, cette fonction nécessitera de donner le chemin complet depuis le dossier data
    """
    path_to_image = os.path.abspath(os.path.join(__file__, "../../data/Pets/images/"+image_name + ".jpg"))
    matrix = Image.open(path_to_image)
    matrix = np.array(matrix)
    return matrix

def five_distance(node1, node2) :
    """
    Calcule la distance euclidienne entre deux pixel en dimension 5 (3 pour le RGB et 2 pour la distance dans la matrice)
    input : deux tableaux de taille 5
    output : float, la distance euclidienne
    """
    sum = 0
    for i in range(5) :
        sum += (node1[i]-node2[i])**2
    return sqrt(sum)

def search_nearest_neighbors(matrix, node_x, node_y, radius_search, k=10) :
    """
    Cette fonction consistue le pré-traitement de l'algorithme de segmentation. Il génère pour un pixel la liste de ses k voisins les plus "proches"
    La notion de "proche" ici inclu la distance sur l'image et la similarité de la couleur (utilise la fonction five_distance). A noter qu'on
    se limite autour d'un certain rayon pour limiter les calculs
    Cette fonction utilise également un flou gaussien.
    input : matrix, l'array numpy encodant l'image
            node_x, node_y, les coordonnées du pixel traité
            radius_search, le rayon auquel on se limite pour la recherche
            k, le nombre de voisins que l'on cherche (par défaut 10)
    output : liste à k éléments de la forme : Distance entre les 2 pixels et tuples de leurs coordonnées
    """
    matrix = scipy.ndimage.gaussian_filter(matrix, 1)
    list_neighbor = []
    n = len(matrix)
    m = len(matrix[0])
    for x in range(min(max(0,node_x-radius_search),n), min(max(0,node_x + radius_search),n)) :
        for y in range(min(max(0,node_y-radius_search),m), min(max(0,node_y+radius_search),m)):
            if (n,m) != (node_x,node_y) :
                temp_element = [five_distance([node_x,node_y] + matrix[node_x][node_y].tolist(), [x,y] + (matrix[x][y]).tolist()), (x, y), (node_x, node_y)]
                list_neighbor.insert(bisect.bisect(list_neighbor, temp_element), temp_element)
            if len(list_neighbor) > k : #On ne garde que k arêtes
                list_neighbor = list_neighbor[:k]
    return list_neighbor

def MInt(unionFind1, unionFind2, k) :
    #Renvoie la fonction MInt décrite dans l'article de référence. Sert à estimer l'uniformité des éléments UnionFind considérés
    return min(unionFind1.find().maxWeight + k/(unionFind1.find().size), unionFind2.find().maxWeight + k/(unionFind2.find().size))

def obtain_sorted_edges(matrix, k=10) :
    """
    Fait appel à search_nearest_neighbors pour chaque pixel de l'image, on relie ainsi chaque pixel à ses k plus proches voisins et cette
    fonction détermine chaque arête. A noter que cette fonction renvoie la liste d'arêtes triée croissante selon le poids de chaque arête
    (c'est-à-dire la distance entre les pixels)
    input : matrice représentant l'image
            k le nombre de voisins par pixel
    output : liste d'arêtes avec distance et coordonnées des pixels
    """
    n = len(matrix)
    m = len(matrix[0])
    listEdges=[]
#    radius_search = ceil(min(n,m)**(1/4)) #Si on souhaite avoir un rayon qui dépende de la taille de l'image, augmente beaucoup le temps de calcul
    radius_search = 4
    for x in range(n):
        for y in range(m):
            nearest_neighbors = search_nearest_neighbors(matrix, x, y, radius_search, k)
            for element in nearest_neighbors :
                bisect.insort(listEdges, element)
    return listEdges

def graph_segmentation(matrix, numberNeighbors=10, k =100, listEdges = None, power = 1) :
    """
    Cette fonction segmente à proprement parler l'image. A partir de la liste des arêtes elle crée une matrice de structure UnionFind et 
    unie ces structures pour les pixels dans le même cluster. Le paramètre squared correspond à une observation empirique : il semblerait qu'en
    mettant le poids de l'arête au carré au moment de la comparaison, on favorise un faible nombre de plus gros clusters (pas pour une même valeur
    de k bien entendu), ce paramètre permet donc d'experimenter un peu avec ceci.
    input : matrice représentant l'image
            numberNeighbors, le nombre de voisins du graphe
            k, paramètre permettant de contrôler heuristiquement la taille des clusters : plus k est grand plus on favorise les gros clusters
            listEdges, la liste des arêtes si on l'a déjà calculée, sinon la fonction la calcule elle-même
    output : matrice de structure UnionFind
    """
    if listEdges == None :
        listEdges = obtain_sorted_edges(matrix, numberNeighbors)
    matrixVertices = []
    n = len(matrix)
    m = len(matrix[0])
    q = len(listEdges)
    for _ in range(n) : #On produit la matrice d'UnionFind
        temp_row = []
        for _ in range(m) :
            temp_row.append(UnionFind())
        matrixVertices.append(temp_row)
    for x in range(q) :
        vertex1_x, vertex1_y = listEdges[x][1]
        vertex2_x, vertex2_y = listEdges[x][2]
        vertex1 = matrixVertices[vertex1_x][vertex1_y]
        vertex2 = matrixVertices[vertex2_x][vertex2_y]
        weight = listEdges[x][0]
        if weight**power < MInt(vertex1,vertex2,k) : #On fusionne les deux UnionFind seulement si l'arête entre les deux est de faible poids
            vertex1.union(vertex2, weight)
    return matrixVertices

def represent_groups(matrixVertices) :
    """
    Cette fonction prend le résultat de graph_segmentation et représente l'image par une matrice d'entier. Pour chaque structure UnionFind
    on associe un entier différent. Chaque entier correspond donc à un segment différent de l'image
    input : matrice d'UnionFind, sortie de graph_segmentation
    output : matrice d'entiers
    """
    matrixResult = []
    n = len(matrixVertices)
    m = len(matrixVertices[0])
    dic = {}
    count = 0
    for a in range(n) :
        temp_row = []
        for b in range(m) :
            root = matrixVertices[a][b].find()
            if root not in dic :
                 count += 1
                 dic[root] = count
            temp_row.append(dic[root])
        matrixResult.append(temp_row)
    return matrixResult


def obtain_sorted_edges_color(matrix, color_index, k = 10) :
    """
    Cette fonction est similaire à obtain_sorted_edges mais remplace la distance euclidienne en dimension 5 par la distance absolue entre une des 
    trois couleurs des pixels. Encore une fois la liste est triée selon la distance considérée
    input : matrice représentant l'image
            color_index, entier entre 0 et 2 donnant le numéro de la couleur qu'on considère
            k, le nombre de voisins
    output : liste d'arêtes avec distance et coordonnées des pixels
    """
    n = len(matrix)
    m = len(matrix[0])
    listEdges=[]
#    radius_search = ceil(min(n,m)**(1/4)) #Si on souhaite avoir un rayon qui dépende de la taille de l'image, augmente beaucoup le temps de calcul
    radius_search = 1
    for x in range(n):
        for y in range(m):
            nearest_neighbors = search_nearest_neighbors(matrix, x, y, radius_search, k)
            for element in nearest_neighbors :
                temp = element[1:]
                x1, y1 = element[1]
                x2, y2 = element[2]
                #distance_colour = abs(matrix[x1][y1][color_index] - matrix[x2][y2][color_index])
                distance_colour = exp(-((matrix[x1][y1][color_index] - matrix[x2][y2][color_index])**2)/200)
                temp.insert(0,distance_colour)
                bisect.insort(listEdges, temp)
    return listEdges

def graph_segmentation_color(matrix, numberNeighbors = 10, k = 100, listsEdges = None, power = 1) :
    """
    Cette fonction st très proche de graph_segmentation à laquelle elle fait appel. Puisqu'on utilise les couleurs cette fois-ci on fait appel
    à graph_segmentation pour chaque couleur puis on évalue l'intersection des clusters pour chaque couleur. Le résultat final est similaire
    à celui de represent_groups
    input : matrix, représente l'image
            numberNeighbors
            k le facteur d'échelle, plus k est grand, plus on favorise les gros clusters
            listsEdges, si les 3 listes d'arêtes ont déjà été calculées on peut les envoyer pour réduire le temps de calcul
    output : matrice d'entiers
    """
    if listsEdges == None :
        listsEdges = []
        for i in range(3) :
            listsEdges.append(obtain_sorted_edges_color(matrix, i, numberNeighbors))
    listMatrixVertices = []
    for listE in listsEdges :
        listMatrixVertices.append(graph_segmentation(matrix, numberNeighbors, k, listE, power = power))
    listMatrixResult = []
    for listMV in listMatrixVertices :
        listMatrixResult.append(represent_groups(listMV))
    matrixFinalResult = []
    n = len(listMatrixResult[0])
    m = len(listMatrixResult[0][0])
    dic = {}
    count = 0
    for a in range(n) :
        temp_row = []
        for b in range(m) :
            tupleColors = (listMatrixResult[0][a][b], listMatrixResult[1][a][b], listMatrixResult[2][a][b])
            if tupleColors not in dic :
                 count += 1
                 dic[tupleColors] = count
            temp_row.append(dic[tupleColors])
        matrixFinalResult.append(temp_row)
    return matrixFinalResult        

def obtain_immediate_neighbors_edges_color(matrix) :
    """
    Cette fonction est similaire aux obtain_sorted_edges, toutefois elle est optimisée pour un rayon de recherche de 1, c'est-à-dire qu'on prend
    pour voisin uniquement les pixels qui sont immédiatement proches (un pixel a 8 voisins proches). Pour ce faire on calcule les arêtes pour
    chaque pixel avec son voisin de droite et ses trois voisins du dessous
    input : matrice représentant l'image
    ouput : liste de 3 liste d'arêtes avec distance et coordonnées des pixels, le tout ordonné selon la distance et une liste pour chaque couleur
    """
    matrixBlur = scipy.ndimage.gaussian_filter(matrix, 1)
    n = len(matrixBlur)
    m = len(matrixBlur[0])
    listEdgesDic = {0 : [], 1 : [], 2 : []}
    for x in range(n-1):
        for y in range(1, m-1):
            for k in range(-1, 2) :
                for color_index in range(3):
                    temp = [abs(int(matrixBlur[x][y][color_index]) - int(matrixBlur[x+1][y+k][color_index])), (x,y), (x+1, y+k)]
                    bisect.insort(listEdgesDic[color_index], temp)
            for color_index in range(3) :
                temp = [abs(int(matrixBlur[x][y][color_index]) - int(matrixBlur[x][y+1][color_index])), (x,y), (x, y+1)]
                bisect.insort(listEdgesDic[color_index], temp)
        for k in range(0,2) :
            for color_index in range(3) :
                temp = [abs(int(matrixBlur[x][0][color_index]) - int(matrixBlur[x+1][k][color_index])), (x,0), (x+1, k)]
                bisect.insort(listEdgesDic[color_index], temp)
                temp = [abs(int(matrixBlur[x][m-1][color_index]) - int(matrixBlur[x+1][m-1-k][color_index])), (x,m-1), (x+1, m-1-k)]
                bisect.insort(listEdgesDic[color_index], temp)
                temp = [abs(int(matrixBlur[x][0][color_index]) - int(matrixBlur[x][1][color_index])), (x,0), (x, 1)]
                bisect.insort(listEdgesDic[color_index], temp)
                temp = [abs(int(matrixBlur[x][m-1][color_index]) - int(matrixBlur[x][m-2][color_index])), (x,m-1), (x, m-2)]
                bisect.insort(listEdgesDic[color_index], temp)
    for y in range(m-2) :
        for color_index in range(3) :
            temp = [abs(int(matrixBlur[n-1][y][color_index]) - int(matrixBlur[n-1][y+1][color_index])), (n-1,y), (n-1, y+1)]
            bisect.insort(listEdgesDic[color_index], temp)
    return [listEdgesDic[0], listEdgesDic[1], listEdgesDic[2]]


def fully_automated_segmentation_color(image_name, k, power = 1, save_to_file = False, immediate_neighbors = True, numberNeighbors = 10) :
    """
    Cette fonction prend en charge tout le processus d'importation et de traitement de l'image. Si save_to_file a une valeur de False (par défaut)
    alors cette fonction renvoie une matrice d'entiers correspondant à l'image traitée. Sinon elle sauvegarde le résultat d'une représentation
    graphique de cette matrice dans un fichier image qui a pour nom : "[nom de l'image en entrée]_[k]_[power].jpg" avec k le facteur d'échelle utilisé
    par l'algorithme et power le facteur utilisé pour la comparaison de l'arête dans la fonction graph_segmentation. Ce fichier est sauvegardé
    dans le dossier ./data/Pets/treated_data/ avec . la racine du projet.
    Note : dans une version ultérieure, cette fonction nécessitera de donner le chemin complet depuis data
    Input : image_name, str
            k, float
            power, float
            save_to_file, bool
    Output : None || matrice d'entier
    """
    mat = import_image(image_name)
    if immediate_neighbors :
        listsEdges = obtain_immediate_neighbors_edges_color(mat)
    else :
        listsEdges = [obtain_sorted_edges_color(mat,0, k = numberNeighbors), obtain_sorted_edges_color(mat,1, k = numberNeighbors), obtain_sorted_edges_color(mat,2, k = numberNeighbors)]
    matrixResult = graph_segmentation_color(mat, numberNeighbors = numberNeighbors, k = k, listsEdges=listsEdges, power = power)
    if save_to_file :
        heatmap(matrixResult)
        plt.savefig(image_name + "_" + str(k) + "_" + str(power) + ".jpg")
    else :
        return matrixResult

def segment_list_of_image(list) :
    for x in list :
        fully_automated_segmentation_color(x, 50000, save_to_file = True)
        plt.show()
        plt.close()
    
import time

def l1_graph_construction(matrix) :
    """
    Cette fonction permet de calculer le graph L1 associé à la matrice. Ce graphe est construit en prenant les
    coefficients obtenus par L1-minimisation. Cette représentation sparse a potentiellement un plus grand potentiel
    discriminatoire que la matrice d'origine avec un algorithme sigma-ball. Pour la L1-minimisation on utilisera
    une fonction de scipy.optimize. Note : Pour référence sur cette méthode, voir SparseRepresentationForComputerVision.pdf
    dans la bibliographie.
    input : Matrice représentant l'image
    output : List d'arêtes avec le poids de l'arête et deux tuples donnant les coordonnées des pixels.
    """

    n = len(matrix)
    m = len(matrix[0])
    D_matrix = [[],[],[]]
    list_edges=[]
    x = time.clock()
    for i in range(n) :
        for j in range(m) :
            D_matrix[0].append(matrix[i][j][0])
            D_matrix[1].append(matrix[i][j][1])
            D_matrix[2].append(matrix[i][j][2])
    D_matrix[0] += [1,0,0]
    D_matrix[1] += [0,1,0]
    D_matrix[2] += [0,0,1]
    alpha_origin = []
    for i in range(len(D_matrix[0])-1) :
        alpha_origin.append([1])
    #for k in range(len(D_matrix[0]) - 3) :
    for k in range(1) :
        temp_D_k_row_0 = D_matrix[0][:k] + D_matrix[0][k+1:]
        temp_D_k_row_1 = D_matrix[1][:k] + D_matrix[1][k+1:]
        temp_D_k_row_2 = D_matrix[2][:k] + D_matrix[2][k+1:]
        temp_D_k = [temp_D_k_row_0, temp_D_k_row_1, temp_D_k_row_2]
        x_k = [[D_matrix[0][k]], [D_matrix[1][k]], [D_matrix[2][k]]]
        k_linear_constraint = scipy.optimize.LinearConstraint(temp_D_k, x_k, x_k)
        alpha_k = scipy.optimize.minimize(np.linalg.norm, alpha_origin, constraints={"type" : "eq", "fun" : lambda alpha: np.linalg.norm(np.matmul(temp_D_k,alpha)-x_k)})
    return alpha_k, time.clock() -x


def calculate_time(name_file) :
    x = time.clock()
    matrix = import_image(name_file)
    listsEdges = obtain_immediate_neighbors_edges_color(matrix)
    temp = time.clock()-x
    matrixResult = graph_segmentation_color(matrix, k = 10000, listsEdges = listsEdges)
    return temp, time.clock()-x
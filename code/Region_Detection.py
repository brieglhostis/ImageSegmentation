import matplotlib.pyplot as plt
import numpy as np
import os.path as path


def extractImage(image_name):
    """
    Return the RGB matrix of the image given
    :param name_image: name of the image
    :return: M RGB-matrix of the image
    """
    path_to_trimap = path.abspath(path.join(sys.argv[0], "../../2a-poleia-segmentationimage/data/Pets/images/"+image_name))
    M = plt.imread(path_to_trimap)

    return M


def Segmentation(img,x,y):
    """
    Define which neighbour pixel will be part of the region of or pixel in x,y
    :param img: RGB matrix of the image
    :param x: x position of our pixel
    :param y: y position of our pixel
    :return: all pixels that should be in the region of our central pixel
    """
    n,m = img.shape

    if x == 0 and y ==0:
        neighbour = [(x,y+1),(x+1,y),(x+1,y+1)]
    elif x==0:
        neighbour = [(x,y-1),(x,y+1)(x+1,y-1),(x+1,y),(x+1,y+1)]
    elif y==0:
        neighbour = [(x-1,y),(x-1,y+1),(x,y+1),(x+1,y),(x+1,y+1)]
    elif x == n-1 and y ==0:
        neighbour = [(x-1,y),(x-1,y+1),(x,y+1)]

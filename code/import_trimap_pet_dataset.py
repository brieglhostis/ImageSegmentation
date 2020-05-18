import os.path as path
import matplotlib.image as img
from seaborn import heatmap
import matplotlib.pyplot as plt

def extract_trimap_dataset(image_name):
    """
    The purpose of this function is to return a matrix with the trimap values of the image that is given in argument
    argument : image_name, a string
    output : a numpy array representing a matrix with values in {1,2,3} (1 is foreground, 2 is edge and 3 background)
    """
    path_to_trimap = path.abspath(path.join(__file__, "../../data/Pets/trimaps/"+image_name))
    matrix = img.imread(path_to_trimap)
    number_line = len(matrix)
    number_column = len(matrix[0])
    for line in range(number_line) :
        for column in range(number_column) :
            if abs(matrix[line][column]-0.00784314) < 0.0001 :
                matrix[line][column] = 3
            if abs(matrix[line][column] - 0.011764706) < 0.0001:
                matrix[line][column] = 2
            if abs(matrix[line][column] - 0.003921569) < 0.0001:
                matrix[line][column] = 1
    return matrix


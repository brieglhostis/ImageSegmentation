import os.path as path
import import_trimap_pet_dataset

def comparing_result_name_pets(result_matrix, image_name) :
    """This function compares how close a matrix that results from a segmentation algorithm to the trimap from the pet database.
    To do so it compares the segments that intersect with the animal segment of the image, and for each of these segments it computes its
    area inside the pet and outside, it then returns number_segments_inside*exterior_area/interior_area. Thus a good algorithm will minimize it
    argument :  result_matrix is a matrix-like array with integer values corresponding to different segments
    image_name, a string corresponding to the name of the image in the dataset
    returns : a float
    """
    trimap_matrix = import_trimap_pet_dataset.extract_trimap_dataset(image_name)
    dictionary_count_inside = {}
    dictionary_count_outside = {}
    number_line = len(trimap_matrix)
    number_columns = len(trimap_matrix[0])
    for line in range(number_line) :
        for column in range(number_columns) :
            if trimap_matrix[line][column] in [1,2] :
                temp = result_matrix[line][column]
                if not temp in dictionary_count_inside :
                    dictionary_count_inside[temp] = 1
                else :
                    dictionary_count_inside[temp] += 1
            else :
                temp = result_matrix[line][column]
                if not temp in dictionary_count_outside :
                    dictionary_count_outside[temp] = 1
                else :
                    dictionary_count_outside[temp] += 1
    sum = 0
    for segment in dictionary_count_inside :
        if segment in dictionary_count_outside :
            sum += dictionary_count_outside[segment]/dictionary_count_inside[segment]
    return sum * len(dictionary_count_inside)
    
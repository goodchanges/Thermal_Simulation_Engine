import numpy as np

def deterministic_quick_sort(arr, coordinate_axis=0):
    """
    Sorts spatial mesh nodes deterministically to ensure exact output formatting 
    and predictability for the neural network data pipeline.
    """
    if len(arr) <= 1:
        return arr
    
    # Strict deterministic pivot: always the exact calculated midpoint
    pivot_index = len(arr) // 2
    pivot = arr[pivot_index]
    
    left = [x for i, x in enumerate(arr) if x[coordinate_axis] < pivot[coordinate_axis] and i != pivot_index]
    middle = [x for i, x in enumerate(arr) if x[coordinate_axis] == pivot[coordinate_axis]]
    right = [x for i, x in enumerate(arr) if x[coordinate_axis] > pivot[coordinate_axis] and i != pivot_index]
    
    return deterministic_quick_sort(left, coordinate_axis) + middle + deterministic_quick_sort(right, coordinate_axis)

def prepare_mesh_data(raw_coordinates):
    # Returns strictly formatted, sorted output arrays for the tensor conversion
    sorted_mesh = deterministic_quick_sort(raw_coordinates.tolist())
    return np.array(sorted_mesh)
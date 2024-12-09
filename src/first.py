import numpy as np
import math
import matplotlib.pyplot as plt
from PIL import Image
from scipy.sparse.linalg import eigs 
from scipy.sparse import diags, dia_matrix

# Number of eigenvectors (energy levels) to calculate
num_energy_levels = 50

# Image filename to define the potential
image_file = 'seedmap.png'

# Range of eigenvectors to generate images for
energy_levels_range = [0, 16]

# Size of the matrix (n x n) for the potential well
# Set n = 30 for faster computation. Larger n increases computation time significantly.
matrix_size = 128 

# Value inside the potential well
inside_well_value = 1

# Value outside the potential well
outside_well_value = 0

# Function to create a polygonal potential well shape
def create_polygonal_shape(sides, radius=1, rotation=0, translation=None):
    segment_angle = math.pi * 2 / sides
    points = [
        (math.sin(segment_angle * i + rotation) * radius,
         math.cos(segment_angle * i + rotation) * radius)
        for i in range(sides)]
    if translation:
        points = [[sum(pair) for pair in zip(point, translation)]
                  for point in points]
    return points

# Function to check if a point is inside a polygon
def is_point_in_polygon(x, y, polygon_x, polygon_y):
    count = 0
    for i in range(len(polygon_x)):
        if (((polygon_y[i] <= y and y < polygon_y[i-1]) or (polygon_y[i-1] <= y and y < polygon_y[i])) and
            (x > (polygon_x[i-1] - polygon_x[i]) * (y - polygon_y[i]) / (polygon_y[i-1] - polygon_y[i]) + polygon_x[i])):
            count = 1 - count
    return count

# Function to create a polygonal potential well matrix
def generate_polygonal_well_matrix(n, inside_value, outside_value):
    well_matrix = np.empty((n, n))
    sides = 6  # Define the number of sides for the polygon
    polygon_points = create_polygonal_shape(sides, 100, 0, (100, 100))
    polygon_x = tuple([polygon_points[i][0] for i in range(0, sides)])
    polygon_y = tuple([polygon_points[i][1] for i in range(0, sides)])

    for i in range(0, n):
        for j in range(0, n):
            if is_point_in_polygon(i, j, polygon_x, polygon_y):
                well_matrix[i][j] = inside_value
            else:
                well_matrix[i][j] = outside_value
    return well_matrix

# Function to create a circular potential well matrix
def generate_circular_well_matrix(n, inside_value, outside_value):
    well_matrix = np.empty((n, n))
    for i in range(0, n):
        for j in range(0, n):
            if math.dist([i, j], [(n-1)/2, (n-1)/2]) <= (n-1)/2:
                well_matrix[i][j] = inside_value
            else:
                well_matrix[i][j] = outside_value
    return well_matrix

# Function to create a double-circle potential well matrix
def generate_bi_circular_well_matrix(n, inside_value, outside_value):
    well_matrix = np.empty((n, n))
    r = n / (2 + math.sqrt(2))
    for i in range(0, n):
        for j in range(0, n):
            if (math.dist([i, j], [r, r]) <= r) or (math.dist([i, j], [n - r - 3, n - r - 3]) <= r):
                well_matrix[i][j] = inside_value
            else:
                well_matrix[i][j] = outside_value
    return well_matrix

# Function to create a toroidal potential well matrix
def generate_toroidal_well_matrix(n, inside_value, outside_value):
    well_matrix = np.empty((n, n))
    r = n / (2 + math.sqrt(2))
    for i in range(0, n):
        for j in range(0, n):
            if (math.dist([i, j], [(n-1)/2, (n-1)/2]) <= (n-1)/2) and (math.dist([i, j], [(n-1)/2, (n-1)/2]) > (n-1)/4):
                well_matrix[i][j] = inside_value
            else:
                well_matrix[i][j] = outside_value
    return well_matrix

# Function to create a rose-shaped potential well matrix
def generate_rose_well_matrix(n, inside_value, outside_value):
    well_matrix = np.empty((n, n))
    sides = 100
    polygonA = create_polygonal_shape(sides, n/2, 0, (n/2 - 10, n/2))
    polygon_x1 = tuple([polygonA[i][0] for i in range(0, sides)])
    polygon_y1 = tuple([polygonA[i][1] for i in range(0, sides)])

    polygonB = create_polygonal_shape(sides, n/2, (2/3) * math.pi, (n/2- 10, n/2))
    polygon_x2 = tuple([polygonB[i][0] for i in range(0, sides)])
    polygon_y2 = tuple([polygonB[i][1] for i in range(0, sides)])

    polygonC = create_polygonal_shape(sides, n/2, -(2/3) * math.pi, (n/2- 10, n/2))
    polygon_x3 = tuple([polygonC[i][0] for i in range(0, sides)])
    polygon_y3 = tuple([polygonC[i][1] for i in range(0, sides)])

    for i in range(0, n):
        for j in range(0, n):
            if is_point_in_polygon(i, j, polygon_x1, polygon_y1) or is_point_in_polygon(i, j, polygon_x2, polygon_y2) or is_point_in_polygon(i, j, polygon_x3, polygon_y3):
                well_matrix[i][j] = inside_value
            else:
                well_matrix[i][j] = outside_value

    well_matrix[int(n/2 - 10)][int(n/2)] = inside_value       
    well_matrix[int(n/2 - 10)][int(n/2 - 1)] = inside_value
    well_matrix[int(n/2 - 10)][int(n/2 + 1)] = inside_value

    return well_matrix

# Function to create a heart-shaped potential well matrix
def generate_heart_well_matrix(n, inside_value, outside_value):
    well_matrix = np.empty((n, n))
    for i in range(0, n):
        for j in range(0, n):
            x = (i - n/2)/110  
            y = (j - n/2 + 20)/110  
            if 5*((x**2 + y**2 - 1)**3) < 6*(x**2)*(y**3):
                well_matrix[i][j] = inside_value
            else:
                well_matrix[i][j] = outside_value
    return well_matrix

# Function to create a seed-shaped potential well matrix from an image
def generate_seed_well_matrix(n, inside_value, outside_value):
    well_matrix = np.empty((n, n))
    image = Image.open(image_file)
    image = np.array(image.convert('1'))
    for i in range(0, n):
        for j in range(0, n):
            if image[i][j]:
                well_matrix[i][j] = inside_value
            else:
                well_matrix[i][j] = outside_value
    return well_matrix

# General function to compute the Hamiltonian matrix for the potential
def compute_hamiltonian(matrix_2d_potential):
    flattened_potential = np.matrix.flatten(matrix_2d_potential)
    num_points = matrix_size * matrix_size
    increment_value = -1 / (np.linspace(0, 1, matrix_size)[1] ** 2)
    zero_value = 4 / increment_value

    diagonal_matrix_N = [increment_value * flattened_potential[i] for i in range(0, num_points - matrix_size)]
    diagonal_matrix_minus_1 = [increment_value * flattened_potential[i] for i in range(0, num_points - 1)]
    diagonal_matrix_0 = [zero_value for i in range(0, num_points)]
    diagonal_matrix_plus_1 = [increment_value * flattened_potential[i] for i in range(0, num_points - 1)]
    diagonal_matrix_plus_N = [increment_value * flattened_potential[i] for i in range(0, num_points - matrix_size)]

    diagonals = [-matrix_size, -1, 0, 1, matrix_size]
    matrix_values = [diagonal_matrix_N, diagonal_matrix_minus_1, diagonal_matrix_0, diagonal_matrix_plus_1, diagonal_matrix_plus_N]

    hamiltonian_matrix = diags(matrix_values, diagonals, format='dia')
    return hamiltonian_matrix

# Function to solve for the eigenvalues and eigenvectors of the Hamiltonian matrix
def solve_hamiltonian(hamiltonian_matrix):
    eigenvalues, eigenvectors = eigs(hamiltonian_matrix, k=num_energy_levels)
    return eigenvalues, eigenvectors

# Function to display and save the images of the wavefunctions
def visualize_and_save_wavefunctions(eigenvectors):
    for i in range(int(energy_levels_range[0]), int(energy_levels_range[1])):
        fig, ax = plt.subplots(1, 1)
        plot = plt.imshow(np.abs(eigenvectors[:, i].reshape(matrix_size, matrix_size)) ** 2, cmap='nipy_spectral', interpolation='gaussian') 
        plt.setp(ax, xticks=[], yticks=[])
        plt.savefig(f'{i}_wavefunction.png', bbox_inches='tight')

# Main program execution
if __name__ == '__main__':
    potential_matrix = generate_circular_well_matrix(matrix_size, inside_well_value, outside_well_value)
    plt.imshow(potential_matrix)
    plt.show()

    hamiltonian_matrix = compute_hamiltonian(potential_matrix)
    eigenvalues, eigenvectors = solve_hamiltonian(hamiltonian_matrix)

    # Sort the eigenvalues and eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Save the eigenvectors (wavefunctions)
    np.save(f'wavefunctions_{matrix_size}x{matrix_size}_e{num_energy_levels}.npy', eigenvectors)

    visualize_and_save_wavefunctions(eigenvectors)

    print('Computation and visualization complete.')
import numpy as np
import cv2

img = cv2.imread("img.bmp")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

n = 4 
m = 4
codebook_size = 256
pop_size = 65
max_gen = 500


def divide_image(img, n, m):
    image_vectors = []
    for i in range(0, img.shape[0] - n, n):
        for j in range(0, img.shape[1] -m, m):
            image_vectors.append(img[i : i + m, j : j + m])
    image_vectors = np.asarray(image_vectors).astype(int)
    return image_vectors


def build_codebook(codebook_size, vector_dimension, image_vectors):
    codebook = np.zeros((codebook_size, vector_dimension)).astype(int)
    for i in range(0, codebook_size):
        m = np.random.randint(0, image_vectors.shape[0])
        codebook[i] = image_vectors[m].reshape(1, vector_dimension)
       
    return codebook

def initialize_population(codebook, pop_size, image_vectors, chromosom_size):
    population = np.zeros((pop_size, chromosom_size))
    for i in range(0, pop_size):
        m = np.random.randint(0, image_vectors.shape[0])
        population[i] = image_vectors[m].reshape(1, chromosom_size)

    return population



image_vectors = divide_image(img, n, m)
codebook = build_codebook(codebook_size, n * m, image_vectors)
population = initialize_population(codebook, pop_size, image_vectors, n * m)
print (population.shape)

#for i in range(0, codebook_size):
    
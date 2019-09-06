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

def psnr(x, y):
    mse = np.square(x - y).mean()
    if mse == 0:
        return -1
    return 10 * np.log10(255 * 255 / mse)
    

def image_block_classifier(image_vectors, codebook):
    index_vector = np.zeros((image_vectors.shape[0],1))
    for i in range(0, image_vectors.shape[0]):
        hpsnr = 0
        best_unit = 1
        for j in range(0, codebook.shape[0]):
            tpsnr = psnr(image_vectors[i].reshape(1, codebook.shape[1]), codebook[j])
            if (tpsnr > hpsnr):
                best_unit = j
                hpsnr = tpsnr
        index_vector[i] = best_unit
    return index_vector

def initialize_population(codebook, pop_size, image_vectors, chromosom_size, index_vector, unit):
    population = np.zeros((pop_size, chromosom_size))
    all_blocks = np.asarray(index_vector == unit).nonzero()[0]
    for i in range(0, pop_size):
        for j in range(0, chromosom_size):
            m = np.random.randint(0,all_blocks.shape[0])
            vector = image_vectors[m].reshape(1, chromosom_size)
            population[i, j] = vector[0][j]

    return population



    

image_vectors = divide_image(img, n, m)
codebook = build_codebook(codebook_size, n * m, image_vectors)
index_vector = image_block_classifier(image_vectors, codebook)
for i in range(0, codebook_size):
    population = initialize_population(codebook, pop_size, image_vectors, n * m, index_vector, i)

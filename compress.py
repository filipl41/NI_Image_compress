import numpy as np
import cv2

img = cv2.imread("img.bmp")
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

n = 4 
m = 4
codebook_size = 256
pop_size = 65
max_gen = 500
mutation_prob = 0.1

def divide_image(img, n, m):
    image_vectors = []
    for i in range(0, img.shape[0] - n + 1, n):
        for j in range(0, img.shape[1] -m + 1, m):
            image_vectors.append(img[i : i + m, j : j + m])
    image_vectors = np.asarray(image_vectors).astype(int)
    return image_vectors


def decode_codebook(codebook, codebook_size, index_vector,  n, m):
    img = np.zeros((512, 512))
    i = 0
    j = 0
    for k in range (0, index_vector.shape[0]):
        index = index_vector[k]
        img[i : i + n, j : j + m] = codebook[index].reshape(4, 4)
        j = j + m
        if (j == 512):
            j = 0
            i = i + n
    return img


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
    return int (10 * np.log10(255 * 255 / mse))
    

def image_block_classifier(image_vectors, codebook):
    index_vector = np.zeros((image_vectors.shape[0],1)).astype(int)
    for i in range(0, image_vectors.shape[0]):
        hpsnr = 0
        best_unit = 0
        for j in range(0, codebook.shape[0]):
            tpsnr = psnr(image_vectors[i].reshape(1, codebook.shape[1]), codebook[j])
            if (tpsnr > hpsnr):
                best_unit = j
                hpsnr = tpsnr
        index_vector[i] = best_unit
    return index_vector

def initialize_population(codebook, pop_size, image_vectors, chromosom_size, index_vector, unit):
    population = np.zeros((pop_size, chromosom_size)).astype(int)
    all_blocks = np.asarray(index_vector == unit).nonzero()[0]
    if (all_blocks.shape[0] == 0):
        return  population
    for i in range(0, pop_size):
        for j in range(0, chromosom_size):
            m = np.random.randint(0,all_blocks.shape[0])
            n_r = all_blocks[m]
            vector = image_vectors[n_r].reshape(1, chromosom_size)
            population[i, j] = vector[0][j]

    return population


def calculate_diff_pixels(image_vectors, index_vector, population, pop_size, chromosom_size, unit):
    diff_pixels = np.zeros((pop_size, chromosom_size))
    arr = np.asarray(index_vector == unit).nonzero()[0]
    for i in range(0, pop_size):
        for j in range(0, chromosom_size):
            value = 0
            for k in range(0, arr.shape[0]):
                ind = arr[k]
                vector = image_vectors[ind].reshape(1, chromosom_size)
                value = value + abs(population[i, j] - vector[0][j])
            diff_pixels[i, j] = value
    return diff_pixels


def fitness_calculation(pop_size, population, chromosom_size, diff_pixel):
    fitness = []
    for i in range(0, pop_size):
        counter = 0
        for j in range(0, pop_size):
            if (i == j):
                continue
            for k in range(0, chromosom_size):
                if diff_pixel[i][k] < diff_pixel[j][k]:
                    counter = counter + 1
        fitness.append(counter / chromosom_size)
    
    fitness =  np.asarray(fitness).astype(float)
    return fitness
    

def selection(fitnes_values):
    parent1 = np.argmax(fitnes_values)
    tmp = np.max(fitnes_values)
    fitnes_values[parent1] = -1
    parent2 = np.argmax(fitnes_values)
    fitnes_values[parent1] = tmp
    return parent1, parent2
        

def crossover(parent1, parent2, codebook_size):
    n = np.random.randint(0, codebook_size)
    children1 = parent1
    children2 = parent2
    children1[ : n] = parent1[ : n]
    children1[n : ] = parent2[n :]
    children2[ : n] = parent2[ : n]
    children2[n : ] = parent1[n : ]
    return children1, children2

def mutation(p, chromosom):
    p_p = np.random.rand()
    if p_p > p:
        m = np.random.randint(0, 16)
        chromosom[m] = np.average(chromosom)

    return chromosom
 
def check_termination_criterion(generation, max_gen, sum_fit, old_fitness, eps):
    if generation >= max_gen:
        return True
    new_average_fit = sum_fit / generation
    old_average_fit = old_fitness / (generation - 1)
    if abs(new_average_fit - old_average_fit) < eps:
        return True
    return False

image_vectors = divide_image(img, n, m)
codebook = build_codebook(codebook_size, n * m, image_vectors)
index_vector = image_block_classifier(image_vectors, codebook)
for i in range(0, codebook_size):
    generation = 0
    termination_criteria = False
    old_fitness = 0
    sumFit = 0
    best_chromosom = np.zeros((1, n * m), dtype=int)
    population = initialize_population(codebook, pop_size, image_vectors, n * m, index_vector, i)
    diff_pixels = calculate_diff_pixels(image_vectors, index_vector,population, pop_size, n * m, i)
    fitness_values = fitness_calculation(pop_size, population, n * m, diff_pixels)
        
    while termination_criteria == False:        
        parent1, parent2 = selection(fitness_values)
        children1, children2 = crossover(population[parent1], population[parent2],n * m)
        chromosom1 = mutation(mutation_prob, children1)
        chromosom2 = mutation(mutation_prob, children2)
        diff_pixels_offspring = calculate_diff_pixels(image_vectors, index_vector, np.vstack((chromosom1, chromosom2)), 2, n * m, i)
        fitness_values_offspring = fitness_calculation(2, np.vstack((chromosom1, chromosom2)), n * m, diff_pixels)
        first = np.random.randint(0, pop_size)
        second = np.random.randint(0, pop_size)
        third = np.random.randint(0, pop_size)
        min_index = first
        if fitness_values[first] > fitness_values[second]:
            min_index = second
        if fitness_values[min_index] > fitness_values[third]:
            min_index = third
        if fitness_values[min_index] < np.amax(fitness_values_offspring):
            arg = np.argmax(fitness_values_offspring)
            if arg == 0:
                population[min_index] = chromosom1
            else :
                population[min_index] = chromosom2

        sumFit = sumFit + np.amax(fitness_values_offspring)
        generation = generation + 1
        if generation == 1:
            old_fitness = sumFit
            continue
        termination_criteria = check_termination_criterion(generation, max_gen, sumFit, old_fitness, 0.01)
        if fitness_values[parent1] > np.amax(fitness_values_offspring):
            best_chromosom = population[parent1]
        else:
            arg = np.argmax(fitness_values_offspring)
            if arg == 0:
                best_chromosom = chromosom1
            else:
                best_chromosom = chromosom2
        old_fitness = sumFit

    codebook[i] = best_chromosom  

img = decode_codebook(codebook, codebook_size, index_vector, n, m)
cv2.imwrite("compressed.bmp", img)

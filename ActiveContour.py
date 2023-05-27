#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 09:20:39 2023

@author: kingstonkishanthan

source: V. Monaco, "Active-Contour," GitHub. [Online]. Available: https://github.com/vmonaco/active-contour. [Accessed: Apr. 11, 2023].
"""

import os
import sys
import cv2 as cv2
from scipy.ndimage import sobel
#import scipy.signal as signal
import numpy as np
import pylab as plb
import matplotlib.cm as cm
from PIL.Image import *
from itertools import product
import matplotlib.pyplot as plt



# Coefficients:
#ALPHA = 10 # C'est pour contrôler la régularité du contour
ALPHA = 0.1
#BETA = 20 # C'est pour contrôler l'attraction du contour aux bords de l'image
BETA = 10
W_LINE = 30 # Contribution de ligne
W_EDGE = 20 # Contribution de bord
#MIN_DISTANCE = 10
MIN_DISTANCE = 5
#INITIAL_SMOOTH = 15
INITIAL_SMOOTH = 35
#INITIAL_ITERATIONS = 30
INITIAL_ITERATIONS = 300
#ITERATIONS_DELTA = 5 
ITERATIONS_DELTA = 5
SMOOTH_FACTOR_DELTA = 4

#NUM_NEIGHBORS = 9
NUM_NEIGHBORS = 10
# Snaxels: pixels d'un contour actif
MAX_SNAXELS = 10000 # On définit un nombre maximum de pixels
INITIAL_DISTANCE_BETWEEN_SNAXELS = 50


# Cette fonction permet d etracer les points en niveau de gris
def _display(image, snaxels=None):
    plt.clf()
    if snaxels is not None:
        for s in snaxels:
            #plt.plot([snaxels[s][0],snaxels[s+1][0]],[snaxels[s][1],snaxels[s+1][1]],'g')
            #plt.plot(s[0],s[1],'g-')
            plt.plot(s[0],s[1],'r.',markersize=8.0)
    
    plt.imshow(image, cmap=cm.Greys_r)
    plt.draw()
    
    return


# Cette foncyion permet de calculer le gradient de l'image
def _gradientImage(image):

    gradient = np.sqrt(sobel(image, 0)**2 + sobel(image, 1)**2)
    gradient -= gradient.min()

    return gradient 



# Cette fonction permet de trouver la position des points
def _inBounds(image, point):
    return np.all(point < np.shape(image)) and np.all(point > 0)



# Cette fonction calcule l'énergie externe du point en utilisant la variation de l'intensité 
# de pixel ans l'image originale et la variation de l'intensité de pixel dans l'image lisée.
# Énergie externe: Correspondance du contour avec l'image
def _externalEnergy(image, smooth_image, point):
    pixel = 255 * image[point[1]][point[0]] # Nous multiplions par 255 pour la normalisation
    smooth_pixel = 255 * smooth_image[point[1]][point[0]]
    # Énergie externe
    external_energy = (W_LINE * pixel) - (W_EDGE * (smooth_pixel**2))
    return external_energy


# Cette fonction calcule l'énergie totale.
def _energy(image, smooth_image, current_point, next_point, previous_point=None):
    # Nous calculons ici la distance euclidienne au carré entre le point actuel et le point suivant
    d_squared = np.linalg.norm(next_point -current_point)**2
    
    if previous_point is None:
        # Nous calculons ici l'énergie utilisant alpha et l'énergie externe
        e =  ALPHA * d_squared + _externalEnergy(image, smooth_image, current_point)
        return e 
    else:
        # Nous calculons ici la dérivée seconde
        deriv = np.sum((next_point - 2 * current_point + previous_point)**2)
        e = 0.5 * (ALPHA * d_squared + BETA * deriv + _externalEnergy(image, smooth_image, current_point))
        return e



# Cette fonction calcule le spositions du contour optimales en minimisant l'énergie pour chaque snaxel
def _iterateContour(image, smooth_image, snaxels, energy_matrix, position_matrix, neighbors):
    snaxels_added = len(snaxels)
    for curr_idx in range(snaxels_added - 1, 0, -1):
        energy_matrix[curr_idx][:][:] = float("inf") # J'initialise les énergies à l'infini
        prev_idx = (curr_idx - 1) % snaxels_added
        next_idx = (curr_idx + 1) % snaxels_added
        
        # Ici, nous calculons la distance entre deux noeuds
        for j, next_neighbor in enumerate(neighbors):
            next_node = snaxels[next_idx] + next_neighbor
            
            if not _inBounds(image, next_node):
                continue
            
            min_energy = float("inf")
            for k, curr_neighbor in enumerate(neighbors):
                curr_node = snaxels[curr_idx] + curr_neighbor
                distance = np.linalg.norm(next_node - curr_node)
                
                if not _inBounds(image, curr_node) or (distance < MIN_DISTANCE):
                    continue
                
                min_energy = float("inf")
                #### Il faut les mettre si non ça ne marche pas
                min_position_k = 0
                min_position_l = 0
                ###
                for l, prev_neighbor in enumerate(neighbors):
                    prev_node = snaxels[prev_idx] + prev_neighbor
                        
                    if not _inBounds(image, prev_node):
                        continue
                        
                    energy = energy_matrix[prev_idx][k][l] + _energy(image, smooth_image, curr_node, next_node, prev_node)
                    
                    if energy < min_energy:
                        min_energy = energy
                        min_position_k = k
                        min_position_l = l
                
                energy_matrix[curr_idx][j][k] = min_energy
                position_matrix[curr_idx][j][k][0] = min_position_k
                position_matrix[curr_idx][j][k][1] = min_position_l
    
    min_final_energy = float("inf")
    min_final_position_j = 0
    min_final_position_k = 0

    for j in range(NUM_NEIGHBORS):
        for k in range(NUM_NEIGHBORS):
            if energy_matrix[snaxels_added - 2][j][k] < min_final_energy:
                min_final_energy = energy_matrix[snaxels_added - 2][j][k]
                min_final_position_j = j
                min_final_position_k = k

    pos_j = min_final_position_j
    pos_k = min_final_position_k
    
    for i in range(snaxels_added - 1, -1, -1):
        snaxels[i] = snaxels[i] + neighbors[pos_j]
        if i > 0:
            pos_j = position_matrix[i - 1][pos_j][pos_k][0]
            pos_k = position_matrix[i - 1][pos_j][pos_k][1]
            
    return min_final_energy



def activeContour(image, snaxels):
    energy_matrix = np.zeros( (MAX_SNAXELS - 1,NUM_NEIGHBORS, NUM_NEIGHBORS), dtype=np.float32)
    position_matrix = np.zeros( (MAX_SNAXELS - 1, NUM_NEIGHBORS, NUM_NEIGHBORS, 2), dtype=np.int32 )
    neighbors = np.array([[i, j] for i in range(-1, 2) for j in range(-1, 2)])
    min_final_energy_prev = float("inf")
    
    counter = 0
    smooth_factor = INITIAL_SMOOTH 
    iterations = INITIAL_ITERATIONS
    gradient_image = _gradientImage(image)
    smooth_image = cv2.blur(gradient_image, (smooth_factor, smooth_factor))
        
    while True: # jusqu'à ce qu'il n'y a plus de changement pour l'énergie minimale
        counter += 1
        if not (counter % iterations):
            iterations += ITERATIONS_DELTA
            if smooth_factor > SMOOTH_FACTOR_DELTA:
                smooth_factor -= SMOOTH_FACTOR_DELTA            
            smooth_image = cv2.blur(gradient_image, (smooth_factor, smooth_factor))
            print ("Deblur step, smooth factor now: ", smooth_factor)
        
        _display(smooth_image, snaxels)
        min_final_energy = _iterateContour(image, smooth_image, snaxels, energy_matrix, position_matrix, neighbors)
        
        if (min_final_energy == min_final_energy_prev) or smooth_factor < SMOOTH_FACTOR_DELTA:
            print ("Min energy reached at ", min_final_energy)
            print ("Final smooth factor ", smooth_factor)
            break
        else:
            min_final_energy_prev = min_final_energy
            
            
            
# C'est pour le cercle
def _pointsOnCircle(center, radius, num_points=12):
    points = np.zeros((num_points, 2), dtype=np.int32)
    # Nous créeons ici une boucle pour créer les coordonnées de cahque point sur le cercle
    for i in range(num_points):
        theta = float(i)/num_points * (2 * np.pi)
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        p = [x, y]
        points[i] = p
        
    return points


# C'est la fonction princiale qu'on appelle à l'extérieur
def activeContourFromCircle(image_file, center, radius):
    image = plb.imread(image_file)
    if image.ndim > 2:
        image = np.mean(image, axis=2)
    print ("Image size: ", image.shape)
    
    plt.ion() # C'est pour activer le mode intéractif
    plt.figure(figsize=np.array(np.shape(image))/50.)
    
    _display(image)
#    num_points = int((2 * np.pi * radius)/_INITIAL_DISTANCE_BETWEEN_SNAXELS)
    snaxels = _pointsOnCircle(center, radius, 50) # 50 points
    # C'est pour afficher l'image avec les points
    _display(image, snaxels)
    activeContour(image, snaxels)
    

    plt.ioff()
    _display(image, snaxels)
    # C'est pour sauvegarder
    #plb.savefig(os.path.splitext(image_file)[0] + "-contour-result.png")
    plt.show()
    return

import numpy as np

def checkdomain(D):
    if D >= 1: 
        D = 0.99
    elif D <= -1:
        D = -0.99
    return D

# Dimensions for Solo12 (actual dimensions)
coxa_length = 0.062  # Length of the coxa
femur_length = 0.200  # Length of the femur
tibia_length = 0.190  # Length of the tibia

def solve_FR(coord, coxa, femur, tibia):
    D = (coord[1]**2 + coord[2]**2 - coxa**2 + coord[0]**2 - femur**2 - tibia**2) / (2 * tibia * femur)
    D = checkdomain(D)
    gamma = np.arctan2(np.sqrt(max(0, 1 - D**2)), D)
    tetta = -np.arctan2(-coord[2], coord[1]) - np.arctan2(np.sqrt(max(0, coord[1]**2 + coord[2]**2 - coxa**2)), -coxa)
    alpha = np.arctan2(coord[0], np.sqrt(max(0, coord[1]**2 + coord[2]**2 - coxa**2))) - np.arctan2(tibia * np.sin(gamma), femur + tibia * np.cos(gamma))
    angles = np.array([tetta, alpha + np.pi / 4., gamma - np.pi / 2.])
    return angles

def solve_FL(coord, coxa, femur, tibia):
    D = (coord[1]**2 + coord[2]**2 - coxa**2 + coord[0]**2 - femur**2 - tibia**2) / (2 * tibia * femur)
    D = checkdomain(D)
    gamma = np.arctan2(np.sqrt(max(0, 1 - D**2)), D)
    tetta = -np.arctan2(-coord[2], coord[1]) - np.arctan2(np.sqrt(max(0, coord[1]**2 + coord[2]**2 - coxa**2)), coxa)
    alpha = np.arctan2(coord[0], np.sqrt(max(0, coord[1]**2 + coord[2]**2 - coxa**2))) - np.arctan2(tibia * np.sin(gamma), femur + tibia * np.cos(gamma))
    angles = np.array([tetta, alpha + np.pi / 4., gamma - np.pi / 2.])
    return angles

def solve_BR(coord, coxa, femur, tibia):
    D = (coord[1]**2 + coord[2]**2 - coxa**2 + coord[0]**2 - femur**2 - tibia**2) / (2 * tibia * femur)
    D = checkdomain(D)
    gamma = np.arctan2(-np.sqrt(max(0, 1 - D**2)), D)
    tetta = -np.arctan2(-coord[2], coord[1]) - np.arctan2(np.sqrt(max(0, coord[1]**2 + coord[2]**2 - coxa**2)), -coxa)
    alpha = np.arctan2(coord[0], np.sqrt(max(0, coord[1]**2 + coord[2]**2 - coxa**2))) - np.arctan2(tibia * np.sin(gamma), femur + tibia * np.cos(gamma))
    angles = np.array([tetta, alpha - np.pi / 4., gamma + np.pi / 2.])
    return angles

def solve_BL(coord, coxa, femur, tibia):
    D = (coord[1]**2 + coord[2]**2 - coxa**2 + coord[0]**2 - femur**2 - tibia**2) / (2 * tibia * femur)
    D = checkdomain(D)
    gamma = np.arctan2(-np.sqrt(max(0, 1 - D**2)), D)
    tetta = -np.arctan2(-coord[2], coord[1]) - np.arctan2(np.sqrt(max(0, coord[1]**2 + coord[2]**2 - coxa**2)), coxa)
    alpha = np.arctan2(coord[0], np.sqrt(max(0, coord[1]**2 + coord[2]**2 - coxa**2))) - np.arctan2(tibia * np.sin(gamma), femur + tibia * np.cos(gamma))
    angles = np.array([tetta, alpha - np.pi / 4., gamma + np.pi / 2.])
    return angles



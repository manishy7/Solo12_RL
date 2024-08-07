import numpy as np

def checkdomain(D):
    return np.clip(D, -0.99, 0.99)

def solve_leg(coord, coxa, femur, tibia, is_right_side):
    D = (coord[1]**2 + coord[2]**2 - coxa**2 + coord[0]**2 - femur**2 - tibia**2) / (2 * tibia * femur)
    D = checkdomain(D)
    gamma = np.arctan2(np.sqrt(max(0, 1 - D**2)), D)
    if not is_right_side:
        gamma = -gamma
    tetta = np.arctan2(coord[1], coord[2]) - np.arctan2(np.sqrt(coord[1]**2 + coord[2]**2 - coxa**2), coxa if is_right_side else -coxa)
    alpha = np.arctan2(coord[0], np.sqrt(coord[1]**2 + coord[2]**2 - coxa**2)) - np.arctan2(tibia * np.sin(gamma), femur + tibia * np.cos(gamma))
    alpha_offset = np.pi / 4 if is_right_side else -np.pi / 4
    gamma_offset = -np.pi / 2 if is_right_side else np.pi / 2
    angles = np.array([tetta, alpha + alpha_offset, gamma + gamma_offset])
    return angles

def solve_FR(coord, coxa, femur, tibia):
    return solve_leg(coord, coxa, femur, tibia, True)

def solve_FL(coord, coxa, femur, tibia):
    return solve_leg(coord, coxa, femur, tibia, False)

def solve_BR(coord, coxa, femur, tibia):
    return solve_leg(coord, coxa, femur, tibia, True)

def solve_BL(coord, coxa, femur, tibia):
    return solve_leg(coord, coxa, femur, tibia, False)

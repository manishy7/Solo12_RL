import numpy as np
from IK_solver_gym import solve_FR, solve_FL, solve_BR, solve_BL

class robotKinematicsGym:
    def __init__(self):
        self.coxa_length = 0.062
        self.femur_length = 0.2
        self.tibia_length = 0.19

    def solve(self, body_orn, body_pos, body2feet):
        fr_angles = solve_FR(body2feet[0], self.coxa_length, self.femur_length, self.tibia_length)
        fl_angles = solve_FL(body2feet[1], self.coxa_length, self.femur_length, self.tibia_length)
        br_angles = solve_BR(body2feet[2], self.coxa_length, self.femur_length, self.tibia_length)
        bl_angles = solve_BL(body2feet[3], self.coxa_length, self.femur_length, self.tibia_length)

        # Validate angles are within expected ranges
        self.validate_angles(fr_angles, 'FR')
        self.validate_angles(fl_angles, 'FL')
        self.validate_angles(br_angles, 'BR')
        self.validate_angles(bl_angles, 'BL')

        return fr_angles, fl_angles, br_angles, bl_angles, body2feet

    def validate_angles(self, angles, leg_name):
        # Placeholder: Add actual limits for validation
        min_limits = np.array([-np.pi, -np.pi, -np.pi])
        max_limits = np.array([np.pi, np.pi, np.pi])
        
        if not np.all((angles >= min_limits) & (angles <= max_limits)):
            print(f"{leg_name}_angles out of range: {angles}")


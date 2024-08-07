
import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import matplotlib.pyplot as plt

import new_IK_Solver as IK_solver
import new_geometrics as geometrics

#Testing of Terrain 

# Connect to PyBullet
p.connect(p.GUI)
p.createCollisionShape(p.GEOM_PLANE)
p.createMultiBody(0, 0)

# Load the Solo12 robot URDF
solo12 = p.loadURDF("/home/manishyadav/Downloads/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf", 
                    [0, 0, 0.35], useFixedBase=False)

# Add earth-like gravity
p.setGravity(0, 0, -9.81)
p.setRealTimeSimulation(1)

# Point the camera at the robot at the desired angle and distance
p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-30, cameraPitch=-30, cameraTargetPosition=[0.0, 0.0, 0.25])






# Define the target position
target_position = [2.0, 2.0, 0.0]  # Example target position
target_visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
p.createMultiBody(baseVisualShapeIndex=target_visual_shape, basePosition=target_position)

# Create heightfield terrain
NUM_HEIGHTFIELD_ROWS = 556
NUM_HEIGHTFIELD_COLUMNS = 556
heightfieldData = np.random.uniform(-1, 1, NUM_HEIGHTFIELD_ROWS * NUM_HEIGHTFIELD_COLUMNS).astype(np.float32)















# Define joint limits
JOINT_LIMITS = {
    "min": -np.pi / 4,  # -45 degrees
    "max": np.pi / 4    # 45 degrees
}

# PD control gains
KP = 1.59  # Proportional gain
KD = 0.1   # Derivative gain

class robotKinematics:
    def __init__(self):
        self.L = 0.1946  # length of robot body
        self.W = 0.078   # width of robot body
        self.coxa = 0.062   # coxa length
        self.femur = 0.200   # femur length
        self.tibia = 0.190   # tibia length
        
        self.bodytoFR0 = np.array([self.L / 2, -self.W / 2, 0])
        self.bodytoFL0 = np.array([self.L / 2, self.W / 2, 0])
        self.bodytoBR0 = np.array([-self.L / 2, -self.W / 2, 0])
        self.bodytoBL0 = np.array([-self.L / 2, self.W / 2, 0])

        self.bodytoFR4 = np.array([self.L / 2, -self.W / 2, -0.2])
        self.bodytoFL4 = np.array([self.L / 2, self.W / 2, -0.2])
        self.bodytoBR4 = np.array([-self.L / 2, -self.W / 2, -0.2])
        self.bodytoBL4 = np.array([-self.L / 2, self.W / 2, -0.2])

    def solve(self, orn, pos, bodytoFeet):
        bodytoFR4 = np.asarray([bodytoFeet[0, 0], bodytoFeet[0, 1], bodytoFeet[0, 2]])
        bodytoFL4 = np.asarray([bodytoFeet[1, 0], bodytoFeet[1, 1], bodytoFeet[1, 2]])
        bodytoBR4 = np.asarray([bodytoFeet[2, 0], bodytoFeet[2, 1], bodytoFeet[2, 2]])
        bodytoBL4 = np.asarray([bodytoFeet[3, 0], bodytoFeet[3, 1], bodytoFeet[3, 2]])

        _bodytoFR0 = geometrics.transform(self.bodytoFR0, orn, pos)
        _bodytoFL0 = geometrics.transform(self.bodytoFL0, orn, pos)
        _bodytoBR0 = geometrics.transform(self.bodytoBR0, orn, pos)
        _bodytoBL0 = geometrics.transform(self.bodytoBL0, orn, pos)

        FRcoord = bodytoFR4 - _bodytoFR0
        FLcoord = bodytoFL4 - _bodytoFL0
        BRcoord = bodytoBR4 - _bodytoBR0
        BLcoord = bodytoBL4 - _bodytoBL0

        undoOrn = -orn
        undoPos = -pos
        _FRcoord = geometrics.transform(FRcoord, undoOrn, undoPos)
        _FLcoord = geometrics.transform(FLcoord, undoOrn, undoPos)
        _BRcoord = geometrics.transform(BRcoord, undoOrn, undoPos)
        _BLcoord = geometrics.transform(BLcoord, undoOrn, undoPos)

        FR_angles = IK_solver.solve_FR(_FRcoord, self.coxa, self.femur, self.tibia)
        FL_angles = IK_solver.solve_FL(_FLcoord, self.coxa, self.femur, self.tibia)
        BR_angles = IK_solver.solve_BR(_BRcoord, self.coxa, self.femur, self.tibia)
        BL_angles = IK_solver.solve_BL(_BLcoord, self.coxa, self.femur, self.tibia)

        if any(np.isnan(angle) for angle in FR_angles + FL_angles + BR_angles + BL_angles):
            return None  # If any angle is NaN, return None

        return FR_angles, FL_angles, BR_angles, BL_angles

class Solo12Env(gym.Env):
    def __init__(self, render=False):
        super(Solo12Env, self).__init__()
        self.render = render
        self.dt = 0.01  # Time step

        # Connect to PyBullet
        self.physicsClient = p.connect(p.GUI if self.render else p.DIRECT)
        
        self.body_id = None
        self.joint_ids = None
        self.robotKinematics = robotKinematics()
        self.last_base_vel = np.zeros(3)
        self.initial_position = np.array([0.0, 0.0, 0.35])
        self.target_position = np.array([10.0, 0.0, 0.35])  # Far target position
        self.reset()

        # Define action and observation space
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(12,), dtype=np.float32)
        obs_len = 3 + 4 + 3 + 3 + 12 * 2 + 2  # 3 (pos) + 4 (orn) + 3 (linear_vel) + 3 (angular_vel) + 24 (12 positions + 12 velocities) + 2 (ZMP_x, ZMP_y)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

    def robot_init(self):
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(
            fixedTimeStep=self.dt,
            numSolverIterations=100,
            enableFileCaching=0,
            numSubSteps=1,
            solverResidualThreshold=1e-10,
            erp=1e-1,
            contactERP=1e-1,
            frictionERP=1e-1
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        body_id = p.loadURDF("/home/manishyadav/Downloads/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf", [0, 0, 0.35], useFixedBase=False)
        
        # Collect joint indices for non-fixed joints
        joint_ids = []
        for j in range(p.getNumJoints(body_id)):
            joint_info = p.getJointInfo(body_id, j)
            if joint_info[2] != p.JOINT_FIXED:
                joint_ids.append(j)
        
        # Set initial joint positions for the legs
        initial_joint_positions = [0, 0.7, -1.4, 0, 0.7, -1.4, 0, 0.7, -1.4, 0, 0.7, -1.4]  # Example positions to stand upright
        for i, joint_id in enumerate(joint_ids):
            p.resetJointState(body_id, joint_id, initial_joint_positions[i])




        # Different Shapes and Terrain options here ------

        # Define the colors for the rectangles
        colors = [
            [1, 0, 0, 1],  # Red
            [0, 0, 1, 1],  # Blue
            [1, 1, 0, 1],  # Yellow
            [0, 1, 0, 1]   # Green
        ]

        # Define the positions for the rectangles
        positions = [
            [-1.0, 0, 0.0],
            [-3.0, 0.5, 0.0],
            [-4.0, -1.0, 0.0],
            [-5.0, 0.5, 0.0]
        ]

        # Define the dimensions for the rectangles (half extents)
        half_extents = [0.1, 0.01, 0.8]

        # Define the orientations for the rectangles in Euler angles
        orientations = [
            [np.pi/2, 0, 0],  # No rotation
            [np.pi/2, 0, 0],  # 90 degrees around X-axis
            [np.pi/2, 0, 0],  # 90 degrees around X-axis
            [np.pi/2, 0, 0]   # 90 degrees around X-axis
        ]

        for color, position, orientation in zip(colors, positions, orientations):
            visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color)
            collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_extents)
            quaternion = p.getQuaternionFromEuler(orientation)
            p.createMultiBody(baseMass=1, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=position, baseOrientation=quaternion)


        
        return body_id, joint_ids

    def reset(self, initial_position=None, target_position=None, **kwargs):
        p.resetSimulation()
        self.body_id, self.joint_ids = self.robot_init()
        self.last_action = np.zeros(12)
        self.last_reward = 0
        self.done = False
        self.steps = 0

        # Update initial and target positions if provided
        if initial_position is not None:
            self.initial_position = np.array(initial_position)
        if target_position is not None:
            self.target_position = np.array(target_position)

        # Initial foot positions
        self.body_to_feet0 = np.array([[0.1946 / 2, -0.15 / 2, -0.2],
                                       [0.1946 / 2, 0.15 / 2, -0.2],
                                       [-0.1946 / 2, -0.15 / 2, -0.2],
                                       [-0.1946 / 2, 0.15 / 2, -0.2]])
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.body_id, self.initial_position, initial_orientation)
        self.last_base_vel = np.zeros(3)  # Reset the last base velocity
        return self.get_observation(), {}

    def get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.body_id)
        joint_states = p.getJointStates(self.body_id, self.joint_ids)
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])

        # Calculate ZMP
        ZMP_x, ZMP_y = self.calculate_zmp(pos, linear_vel)

        observation = np.concatenate([pos, orn, linear_vel, angular_vel, joint_positions, joint_velocities, [ZMP_x, ZMP_y]])

        if np.any(np.isnan(observation)):
            observation = np.nan_to_num(observation)
            print("Warning: NaN values found in observation. Replacing NaNs with zeros.")

        return observation

    def apply_joint_limits(self, angles):
        return np.clip(angles, JOINT_LIMITS["min"], JOINT_LIMITS["max"])

    def pd_control(self, target_positions, current_positions, current_velocities):
        position_error = target_positions - current_positions
        velocity_error = -current_velocities
        torques = KP * position_error + KD * velocity_error
        return torques

    def calculate_zmp(self, base_pos, base_vel):
        base_acc = (np.array(base_vel) - self.last_base_vel) / self.dt
        self.last_base_vel = np.array(base_vel)

        mass = 2.5  # Mass of the robot
        gravity = 9.81

        F_z = mass * gravity + base_acc[2]

        M_x = mass * (base_acc[1] * base_pos[2] - base_acc[2] * base_pos[1])
        M_y = mass * (base_acc[0] * base_pos[2] - base_acc[2] * base_pos[0])

        if F_z == 0:
            ZMP_x = 0
            ZMP_y = 0
        else:
            ZMP_x = -M_y / F_z
            ZMP_y = M_x / F_z

        return ZMP_x, ZMP_y

    def simple_gait(self, step_size):
        # Simple gait pattern for walking forward
        foot_positions = self.body_to_feet0.copy()
        foot_positions[:, 0] += step_size
        return foot_positions

    def step(self, action):
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        pos = np.array(pos)
        orn = np.array(p.getEulerFromQuaternion(orn))

        step_size = 0.01 

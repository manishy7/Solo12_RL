import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from maz_Kinematic_Model_v2 import robotKinematics
from mz_Gait_Planner import trotGait

# Define joint limits (example values, adjust as needed)
JOINT_LIMITS = {
    "min": -np.pi / 4,  # -45 degrees
    "max": np.pi / 4    # 45 degrees
}

# PD control gains
KP = 1.59  # Proportional gain # 1.59
KD = 0.1   # Derivative gain #0.1

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
        self.trot = trotGait()
        self.last_base_vel = np.zeros(3)
        self.target_position = np.array([5.0, 0.0, 0.35])  # Example target position
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
        joint_ids = [j[0] for j in [p.getJointInfo(body_id, j) for j in range(p.getNumJoints(body_id))] if j[2] != p.JOINT_FIXED]
        
        # Set initial joint positions for the legs
        initial_joint_positions = [0, 0.7, -1.4] * 4  # Example positions to stand upright
        for i, joint_id in enumerate(joint_ids):
            p.resetJointState(body_id, joint_id, initial_joint_positions[i])
        
        return body_id, joint_ids

    def reset(self, **kwargs):
        p.resetSimulation()
        self.body_id, self.joint_ids = self.robot_init()
        self.last_action = np.zeros(12)
        self.last_reward = 0
        self.done = False
        self.steps = 0

        # Initial foot positions
        self.body_to_feet0 = np.array([[0.1946 / 2, -0.15 / 2, -0.2],
                                       [0.1946 / 2, 0.15 / 2, -0.2],
                                       [-0.1946 / 2, -0.15 / 2, -0.2],
                                       [-0.1946 / 2, 0.15 / 2, -0.2]])
        initial_position = [0, 0, 0.35]  # Ensure it's above ground
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.body_id, initial_position, initial_orientation)
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
        torques = KP * position_error + KD * velocity_error #Implement -----
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

    def step(self, action):
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        pos = np.array(pos)
        orn = np.array(p.getEulerFromQuaternion(orn))

        L, angle, Lrot, T, sda, offset = 0.05, 0, 0, 0.5, 0, np.array([0.5, 0.5, 0, 0])
        body_to_feet = self.trot.loop(L, angle, Lrot, T, offset, self.body_to_feet0, sda)

        angles_result = self.robotKinematics.solve(orn, pos, body_to_feet)
        
        if angles_result is None:
            return self.get_observation(), -1, True, False, {}

        fr_angles, fl_angles, br_angles, bl_angles, _ = angles_result

        if any(np.isnan(angle) for angle in fr_angles + fl_angles + br_angles + bl_angles):
            return self.get_observation(), -1, True, False, {}

        fr_angles = self.apply_joint_limits(fr_angles)
        fl_angles = self.apply_joint_limits(fl_angles)
        br_angles = self.apply_joint_limits(br_angles)
        bl_angles = self.apply_joint_limits(bl_angles)

        target_positions = np.concatenate([fr_angles, fl_angles, br_angles, bl_angles])
        target_positions += action

        joint_states = p.getJointStates(self.body_id, self.joint_ids)
        current_positions = np.array([state[0] for state in joint_states]) # This is a 12x1 vector (joint angles)
        current_velocities = np.array([state[1] for state in joint_states]) # This is a 12x1 vector (joint velocities)
        
        torques = self.pd_control(target_positions, current_positions, current_velocities)
        torques = np.clip(torques, -15.0, 15.0)  # Adjust torque limits

        # Debugging: Print joint angles and torques
        print(f"Current Positions: {current_positions}")
        print(f"Target Positions: {target_positions}")
        print(f"Torques: {torques}")

        for i in range(12):
            p.setJointMotorControl2(self.body_id, self.joint_ids[i], p.TORQUE_CONTROL, force=torques[i])

        p.stepSimulation()
        time.sleep(self.dt)
        self.steps += 1

        observation = self.get_observation()
        reward = self.compute_reward(observation)
        done = self.compute_done(observation)

        if done:
            self.reset()

        return observation, reward, done, False, {}

    def compute_reward(self, observation):
        try:
            pos, orn, linear_vel, angular_vel, joint_positions, joint_velocities, ZMP_x, ZMP_y = np.split(observation, [3, 7, 10, 13, 25, 37, 38])
            # Reward for maintaining balance and moving towards the target
            reward = -np.sum(np.square(pos - self.target_position))  # Distance to target position
            reward -= np.sum(np.square(orn))  # Orientation penalty
            reward -= np.sum(np.square(linear_vel))  # Linear velocity penalty
            reward -= np.sum(np.square(angular_vel))  # Angular velocity penalty
            reward -= np.sum(np.square(ZMP_x)) + np.sum(np.square(ZMP_y))  # ZMP penalty

            return reward
        except Exception as e:
            print("Reward calculation error:", e)
            return -1

    def compute_done(self, observation):
        pos, _, _, _, _, _, _, _ = np.split(observation, [3, 7, 10, 13, 25, 37, 39])
        return np.abs(pos[2]) < 0.1 or np.abs(pos[2]) > 0.5 or np.linalg.norm(pos[:2] - self.target_position[:2]) < 0.1

def train_rl_agent():
    env = DummyVecEnv([lambda: Solo12Env(render=True)])
    model = PPO("MlpPolicy", env, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./models/', name_prefix='solo12_model')
    model.learn(total_timesteps=100000, callback=checkpoint_callback)

if __name__ == "__main__":
    train_rl_agent()


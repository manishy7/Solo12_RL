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
    "min": -np.pi / 4,
    "max": np.pi / 4
}

# PD control gains
KP = 5.0  # Further reduced proportional gain
KD = 0.05   # Further reduced derivative gain

class Solo12Env(gym.Env):
    def __init__(self, render=False):
        super(Solo12Env, self).__init__()
        self.render = render
        self.dt = 0.01  # Increase the time step for more stability
        self.physicsClient = p.connect(p.GUI if self.render else p.DIRECT)
        self.body_id, self.joint_ids = self.robot_init()
        self.robotKinematics = robotKinematics()
        self.trot = trotGait()
        self.reset()

        # Define action and observation space
        self.action_space = spaces.Box(low=-0.02, high=0.02, shape=(12,), dtype=np.float32)  # Further reduced action range
        obs_len = 3 + 4 + 3 + 3 + 12 * 2  # 12 joints
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
            contactERP=0.0,
            frictionERP=0.0,
        )
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        body_id = p.loadURDF("/home/manishyadav/Downloads/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf", [0, 0, 0.35], useFixedBase=False)
        joint_ids = [p.getJointInfo(body_id, j) for j in range(p.getNumJoints(body_id))]
        return body_id, joint_ids

    def reset(self, **kwargs):
        p.resetSimulation()
        self.body_id, self.joint_ids = self.robot_init()
        self.last_action = np.zeros(12)
        self.last_reward = 0
        self.done = False
        self.steps = 0

        # Initial foot positions
        self.body_to_feet0 = np.array([[0.1946 / 2, -0.15 / 2, 0.1],
                                       [0.1946 / 2, 0.15 / 2, 0.1],
                                       [-0.1946 / 2, -0.15 / 2, 0.1],
                                       [-0.1946 / 2, 0.15 / 2, 0.1]])
        initial_position = [0, 0, 0.35]  # Lower the initial position to reduce initial instability
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])
        p.resetBasePositionAndOrientation(self.body_id, initial_position, initial_orientation)
        return self.get_observation(), {}

    def get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.body_id)
        joint_states = p.getJointStates(self.body_id, range(12))
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        observation = np.concatenate([pos, orn, linear_vel, angular_vel, joint_positions, joint_velocities])

        if np.any(np.isnan(observation)):
            observation = np.nan_to_num(observation)

        return observation

    def apply_joint_limits(self, angles):
        return np.clip(angles, JOINT_LIMITS["min"], JOINT_LIMITS["max"])

    def pd_control(self, target_positions, current_positions, current_velocities):
        position_error = target_positions - current_positions
        velocity_error = -current_velocities
        torques = KP * position_error + KD * velocity_error
        return torques

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

        joint_states = p.getJointStates(self.body_id, range(12))
        current_positions = np.array([state[0] for state in joint_states])
        current_velocities = np.array([state[1] for state in joint_states])

        torques = self.pd_control(target_positions, current_positions, current_velocities)

        for i in range(12):
            p.setJointMotorControl2(self.body_id, i, p.TORQUE_CONTROL, force=torques[i])

        p.stepSimulation()
        time.sleep(self.dt)
        observation = self.get_observation()
        reward = self.compute_reward(observation)
        self.done = self.compute_done(observation)
        info = {}
        return observation, reward, self.done, False, info

    def compute_reward(self, observation):
        pos, orn, linear_vel, angular_vel, joint_positions, joint_velocities = np.split(observation, [3, 7, 10, 13, 25])
        
        forward_reward = linear_vel[0]
        stability_penalty = np.linalg.norm(angular_vel)
        energy_penalty = np.linalg.norm(self.last_action)

        reward = forward_reward - stability_penalty - energy_penalty
        return reward

    def compute_done(self, observation):
        pos, orn = np.split(observation, [3])
        done = pos[2] < 0.3  # Ensure the robot is considered fallen if it drops below a certain height
        return done

    def close(self):
        p.disconnect()

def train_rl_agent():
    env = DummyVecEnv([lambda: Solo12Env(render=True)])
    model = PPO('MlpPolicy', env, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='solo12_rl_model')
    
    try:
        model.learn(total_timesteps=100000, callback=checkpoint_callback)
    except ValueError as e:
        print("Training interrupted due to an error:", e)
    
    model.save("solo12_rl_model")

if __name__ == '__main__':
    train_rl_agent()

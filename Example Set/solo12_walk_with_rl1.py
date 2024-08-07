import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

from kinematic_model import robotKinematics
from gaitPlanner import trotGait

# Define joint limits (example values, adjust as needed)
JOINT_LIMITS = {
    "min": -np.pi / 6,
    "max": np.pi / 6
}

class Solo12Env(gym.Env):
    def __init__(self, render=False):
        super(Solo12Env, self).__init__()
        self.render = render
        self.dt = 0.005
        self.physicsClient = p.connect(p.GUI if self.render else p.DIRECT)
        self.body_id, self.joint_ids = self.robot_init()
        self.robotKinematics = robotKinematics()
        self.trot = trotGait()
        self.reset()

        # Define action and observation space
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(12,), dtype=np.float32)
        obs_len = 3 + 4 + 3 + 3 + p.getNumJoints(self.body_id) * 2
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
        body_id = p.loadURDF("/home/manishyadav/Downloads/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf", [0, 0, 0.48], useFixedBase=False)
        joint_ids = [p.getJointInfo(body_id, j) for j in range(p.getNumJoints(body_id))]
        return body_id, joint_ids

    def reset(self, **kwargs):
        # Reset the robot to the initial state
        p.resetSimulation()
        self.body_id, self.joint_ids = self.robot_init()
        self.last_action = np.zeros(12)  # Replace with appropriate action space size
        self.last_reward = 0
        self.done = False
        self.steps = 0

        # Initial foot positions
        self.body_to_feet0 = np.array([[0.1946 / 2, -0.15 / 2, 0.1],
                                       [0.1946 / 2, 0.15 / 2, 0.1],
                                       [-0.1946 / 2, -0.15 / 2, 0.1],
                                       [-0.1946 / 2, 0.15 / 2, 0.1]])
        return self.get_observation(), {}

    def get_observation(self):
        # Return the current state of the robot
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.body_id)
        joint_states = p.getJointStates(self.body_id, range(p.getNumJoints(self.body_id)))
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        observation = np.concatenate([pos, orn, linear_vel, angular_vel, joint_positions, joint_velocities])
        return observation

    def apply_joint_limits(self, angles):
        return np.clip(angles, JOINT_LIMITS["min"], JOINT_LIMITS["max"])

    def step(self, action):
        # Apply the gait planner for forward movement
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        pos = np.array(pos)
        orn = np.array(p.getEulerFromQuaternion(orn))  # Convert quaternion to Euler angles

        L, angle, Lrot, T, sda, offset = 0.05, 0, 0, 0.5, 0, np.array([0.5, 0.5, 0, 0])
        body_to_feet = self.trot.loop(L, angle, Lrot, T, offset, self.body_to_feet0, sda)

        fr_angles, fl_angles, br_angles, bl_angles, _ = self.robotKinematics.solve(orn, pos, body_to_feet)

        # Debugging: Print joint angles to verify correctness
        
        # Move joints according to the gait planner with limits
        fr_angles = self.apply_joint_limits(fr_angles)
        fl_angles = self.apply_joint_limits(fl_angles)
        br_angles = self.apply_joint_limits(br_angles)
        bl_angles = self.apply_joint_limits(bl_angles)

        for i in range(3):
            p.setJointMotorControl2(self.body_id, i, p.POSITION_CONTROL, targetPosition=fr_angles[i], force=2)
            p.setJointMotorControl2(self.body_id, 4 + i, p.POSITION_CONTROL, targetPosition=fl_angles[i], force=2)
            p.setJointMotorControl2(self.body_id, 8 + i, p.POSITION_CONTROL, targetPosition=br_angles[i], force=2)
            p.setJointMotorControl2(self.body_id, 12 + i, p.POSITION_CONTROL, targetPosition=bl_angles[i], force=2)

        # Apply the given action to the robot joints for balance with limits
        for i in range(12):
            current_pos = p.getJointState(self.body_id, i)[0]
            target_pos = current_pos + action[i]  # Adjust position based on action
            target_pos = np.clip(target_pos, JOINT_LIMITS["min"], JOINT_LIMITS["max"])  # Apply joint limits
            p.setJointMotorControl2(self.body_id, i, p.POSITION_CONTROL, targetPosition=target_pos, force=10)

        p.stepSimulation()
        time.sleep(self.dt)
        observation = self.get_observation()
        reward = self.compute_reward(observation)
        self.done = self.compute_done(observation)
        info = {}
        return observation, reward, self.done, False, info

    def compute_reward(self, observation):
        pos, orn, linear_vel, angular_vel, joint_positions, joint_velocities = np.split(observation, [3, 7, 10, 13, 13 + p.getNumJoints(self.body_id)])
        
        forward_reward = linear_vel[0]  # Reward forward velocity
        stability_penalty = np.linalg.norm(angular_vel)  # Penalize angular velocity
        energy_penalty = np.linalg.norm(self.last_action)  # Penalize large actions to encourage efficiency

        reward = forward_reward - stability_penalty - energy_penalty
        return reward

    def compute_done(self, observation):
        pos, orn = np.split(observation, [3])
        done = pos[2] < 0.3  # Example condition if robot falls
        return done

    def close(self):
        p.disconnect()

def train_rl_agent():
    env = DummyVecEnv([lambda: Solo12Env(render=True)])
    model = PPO('MlpPolicy', env, verbose=1)
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./logs/', name_prefix='solo12_rl_model')
    model.learn(total_timesteps=100000, callback=checkpoint_callback)
    model.save("solo12_rl_model")

if __name__ == '__main__':
    train_rl_agent()
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
from pybullet_debugger import pybulletDebug  
from gaitPlanner import trotGait

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
        # Actions are the target positions for the joints
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(12,), dtype=np.float32)

        # Observations include position, orientation, linear velocity, angular velocity, joint positions, and joint velocities
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

    def step(self, action):
        self.last_action = action
        self.apply_action(action)
        p.stepSimulation()
        time.sleep(self.dt)
        observation = self.get_observation()
        reward = self.compute_reward(observation)
        self.done = self.compute_done(observation)
        info = {}
        return observation, reward, self.done, False, info

    def apply_action(self, action):
        # Apply the given action to the robot joints
        for i in range(12):
            # Update joint positions based on the action (to be refined)
            p.setJointMotorControl2(self.body_id, i, p.POSITION_CONTROL, targetPosition=action[i], force=2)

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
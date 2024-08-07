import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

class Solo12Env(gym.Env):
    def __init__(self):
        super(Solo12Env, self).__init__()
        
        # Define the number of obstacles
        self.num_obstacles = 5
        
        # Define action and observation space
        # Actions: motor positions for each joint
        self.action_space = spaces.Box(low=-1, high=1, shape=(12,), dtype=np.float32)
        
        # Observations: robot's position, orientation, velocities, joint states, and distances to obstacles
        obs_high = np.array([np.inf] * (3 + 4 + 3 + 3 + 12 + 12 + self.num_obstacles), dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)
        
        # Connect to PyBullet
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # Define the target position
        
        
        # Load the Solo12 robot URDF
        self.solo12 = None
        self.obstacles = []
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.createCollisionShape(p.GEOM_PLANE)
        p.createMultiBody(0, 0)
        self.solo12 = p.loadURDF("/home/manishyadav/Downloads/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf", 
                                 [0, 0, 0.35], useFixedBase=False)
        
        self._setup_environment()
        self.time_step = 0
        return self.get_obs(), {}

    def _setup_environment(self):
        # Camera setup
        p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=-30, cameraPitch=-30, cameraTargetPosition=[0.0, 0.0, 0.25])
        
        # Define the dimensions and offsets
        self.xhipf = 0.2
        self.xhipb = -0.2
        self.yhipl = 0.1
        self.yoffh = 0.05
        self.hu = 0.2
        self.hl = 0.2
        
        self.setlegsxyz([self.xhipf, self.xhipf, self.xhipb, self.xhipb], 
                        [self.yhipl + 0.1, -self.yhipl - 0.1, self.yhipl + 0.1, -self.yhipl - 0.1], 
                        [-0.5, -0.5, -0.5, -0.5], [1, 1, 1, 1])

        # Add obstacles
        self._add_obstacles()

    def _add_obstacles(self):
        self.obstacles = []
        for _ in range(self.num_obstacles):
            pos = [np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0.1]
            obstacle = p.loadURDF("block.urdf", basePosition=pos)
            self.obstacles.append(obstacle)

    def xyztoang(self, x, y, z, yoffh, hu, hl):
        dyz = np.sqrt(y**2 + z**2)
        lyz = np.sqrt(dyz**2 - yoffh**2)
        gamma_yz = -np.arctan(y / z)
        gamma_h_offset = -np.arctan(-yoffh / lyz)
        gamma = gamma_yz - gamma_h_offset

        lxzp = np.sqrt(lyz**2 + x**2)
        n = (lxzp**2 - hl**2 - hu**2) / (2 * hu)
        if np.abs(n / hl) > 1:
            beta = 0
        else:
            beta = -np.arccos(n / hl)

        alfa_xzp = -np.arctan(x / lyz)
        alfa_off = np.arccos((hu + n) / lxzp) if np.abs((hu + n) / lxzp) <= 1 else 0
        alfa = alfa_xzp + alfa_off

        if any(np.isnan([gamma, alfa, beta])):
            print(x, y, z, yoffh, hu, hl)
        return [gamma, alfa, beta]

    def setlegsxyz(self, xvec, yvec, zvec, vvec):
        leg_joints = [[0, 1, 2], [4, 5, 6], [8, 9, 10], [12, 13, 14]]
        a = self.xyztoang(xvec[0] - self.xhipf, yvec[0] - self.yhipl, zvec[0], self.yoffh, self.hu, self.hl)
        spd = 1
        p.setJointMotorControl2(self.solo12, leg_joints[0][0], p.POSITION_CONTROL, targetPosition=a[0], force=1000, maxVelocity=spd)
        p.setJointMotorControl2(self.solo12, leg_joints[0][1], p.POSITION_CONTROL, targetPosition=a[1], force=1000, maxVelocity=vvec[0])
        p.setJointMotorControl2(self.solo12, leg_joints[0][2], p.POSITION_CONTROL, targetPosition=a[2], force=1000, maxVelocity=vvec[0])

        a = self.xyztoang(xvec[1] - self.xhipf, yvec[1] + self.yhipl, zvec[1], -self.yoffh, self.hu, self.hl)
        p.setJointMotorControl2(self.solo12, leg_joints[1][0], p.POSITION_CONTROL, targetPosition=a[0], force=1000, maxVelocity=spd)
        p.setJointMotorControl2(self.solo12, leg_joints[1][1], p.POSITION_CONTROL, targetPosition=a[1], force=1000, maxVelocity=vvec[1])
        p.setJointMotorControl2(self.solo12, leg_joints[1][2], p.POSITION_CONTROL, targetPosition=a[2], force=1000, maxVelocity=vvec[1])

        a = self.xyztoang(xvec[2] - self.xhipb, yvec[2] - self.yhipl, zvec[2], self.yoffh, self.hu, self.hl)
        p.setJointMotorControl2(self.solo12, leg_joints[2][0], p.POSITION_CONTROL, targetPosition=a[0], force=1000, maxVelocity=spd)
        p.setJointMotorControl2(self.solo12, leg_joints[2][1], p.POSITION_CONTROL, targetPosition=a[1], force=1000, maxVelocity=vvec[2])
        p.setJointMotorControl2(self.solo12, leg_joints[2][2], p.POSITION_CONTROL, targetPosition=a[2], force=1000, maxVelocity=vvec[2])

        a = self.xyztoang(xvec[3] - self.xhipb, yvec[3] + self.yhipl, zvec[3], -self.yoffh, self.hu, self.hl)
        p.setJointMotorControl2(self.solo12, leg_joints[3][0], p.POSITION_CONTROL, targetPosition=a[0], force=1000, maxVelocity=spd)
        p.setJointMotorControl2(self.solo12, leg_joints[3][1], p.POSITION_CONTROL, targetPosition=a[1], force=1000, maxVelocity=vvec[3])
        p.setJointMotorControl2(self.solo12, leg_joints[3][2], p.POSITION_CONTROL, targetPosition=a[2], force=1000, maxVelocity=vvec[3])

    def get_obs(self):
        position, orientation = p.getBasePositionAndOrientation(self.solo12)
        linear_velocity, angular_velocity = p.getBaseVelocity(self.solo12)
        joint_states = p.getJointStates(self.solo12, range(12))
        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]
        
        # Distances to obstacles
        distances = []
        robot_pos = np.array(position)
        for obstacle in self.obstacles:
            obs_pos, _ = p.getBasePositionAndOrientation(obstacle)
            dist = np.linalg.norm(robot_pos - np.array(obs_pos))
            distances.append(dist)
        
        obs = np.concatenate([position, orientation, linear_velocity, angular_velocity, joint_positions, joint_velocities, distances]).astype(np.float32)
        return obs

    def step(self, action):
        # Apply actions to the joints
        for i in range(12):
            p.setJointMotorControl2(self.solo12, i, p.POSITION_CONTROL, targetPosition=action[i])
        
        p.stepSimulation()
        time.sleep(0.01)
        
        obs = self.get_obs()
        reward = float(self._compute_reward(obs))  # Ensure reward is a float
        done = self._is_done(obs)
        
        terminated = done  # Done condition
        truncated = False  # No specific truncation condition
        
        self.time_step += 1
        return obs, reward, terminated, truncated, {}

    def _compute_reward(self, obs):
        # Reward can be defined based on various factors such as forward velocity, stability, avoiding obstacles, etc.
        position = obs[:3]
        distances = obs[-self.num_obstacles:]
        
        # Reward for forward motion along the x-axis
        forward_reward = position[0]
        
        # Penalty for being close to obstacles
        obstacle_penalty = np.sum(np.exp(-distances))
        
        reward = forward_reward - obstacle_penalty
        return reward

    def _is_done(self, obs):
        # Define a condition to end the episode
        position = obs[:3]
        if position[2] < 0.2:  # End episode if the robot falls
            return True
        if self.time_step > 1000:  # End episode after 1000 time steps
            return True
        return False

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()

# Create and check the custom environment
env = Solo12Env()
check_env(env, warn=True)

# Instantiate the agent
model = PPO('MlpPolicy', env, verbose=1)

# Train the agent
model.learn(total_timesteps=100000)

# Save the model
model.save("ppo_solo12")

# Load the trained model (optional)
# model = PPO.load("ppo_solo12")

# Evaluate the agent
obs, _ = env.reset()
for i in range(1000):
    action, _states = model.predict(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
    time.sleep(0.01)  # Sleep to make the simulation real-time
    env.render()

env.close()


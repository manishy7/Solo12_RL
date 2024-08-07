import time
import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
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

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-0.1, high=0.1, shape=(12,), dtype=np.float32)
        obs_len = 39  # Match the expected length of the model
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)
        
        self.reset()

    def robot_init(self):
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(fixedTimeStep=self.dt, numSolverIterations=100)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        body_id = p.loadURDF("/home/manishyadav/Downloads/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf", [0, 0, 0.48], useFixedBase=False)
        joint_ids = [p.getJointInfo(body_id, j) for j in range(p.getNumJoints(body_id))]
        return body_id, joint_ids

    def reset(self, **kwargs):
        p.resetSimulation()
        self.body_id, self.joint_ids = self.robot_init()
        self.last_action = np.zeros(12)
        self.done = False
        self.steps = 0
        self.body_to_feet0 = np.array([[0.1946 / 2, -0.15 / 2, 0.1],
                                       [0.1946 / 2, 0.15 / 2, 0.1],
                                       [-0.1946 / 2, -0.15 / 2, 0.1],
                                       [-0.1946 / 2, 0.15 / 2, 0.1]])
        return self.get_observation()

    def get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.body_id)
        joint_states = p.getJointStates(self.body_id, range(p.getNumJoints(self.body_id)))
        joint_positions = np.array([state[0] for state in joint_states])
        joint_velocities = np.array([state[1] for state in joint_states])
        observation = np.concatenate([pos, orn, linear_vel, angular_vel, joint_positions, joint_velocities])
        return observation[:39]  # Ensure the observation matches the expected shape

    def apply_joint_limits(self, angles):
        return np.clip(angles, JOINT_LIMITS["min"], JOINT_LIMITS["max"])

    def step(self, action):
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        pos = np.array(pos)
        orn = np.array(p.getEulerFromQuaternion(orn))

        L, angle, Lrot, T, sda, offset = 0.05, 0, 0, 0.5, 0, np.array([0.5, 0.5, 0, 0])
        body_to_feet = self.trot.loop(L, angle, Lrot, T, offset, self.body_to_feet0, sda)
        fr_angles, fl_angles, br_angles, bl_angles, _ = self.robotKinematics.solve(orn, pos, body_to_feet)

        fr_angles = self.apply_joint_limits(fr_angles)
        fl_angles = self.apply_joint_limits(fl_angles)
        br_angles = self.apply_joint_limits(br_angles)
        bl_angles = self.apply_joint_limits(bl_angles)

        for i in range(3):
            p.setJointMotorControl2(self.body_id, i, p.POSITION_CONTROL, targetPosition=fr_angles[i], force=2)
            p.setJointMotorControl2(self.body_id, 4 + i, p.POSITION_CONTROL, targetPosition=fl_angles[i], force=2)
            p.setJointMotorControl2(self.body_id, 8 + i, p.POSITION_CONTROL, targetPosition=br_angles[i], force=2)
            p.setJointMotorControl2(self.body_id, 12 + i, p.POSITION_CONTROL, targetPosition=bl_angles[i], force=2)

        for i in range(12):
            current_pos = p.getJointState(self.body_id, i)[0]
            target_pos = current_pos + action[i]
            target_pos = np.clip(target_pos, JOINT_LIMITS["min"], JOINT_LIMITS["max"])
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
        forward_reward = linear_vel[0]
        stability_penalty = np.linalg.norm(angular_vel)
        energy_penalty = np.linalg.norm(self.last_action)
        reward = forward_reward - stability_penalty - energy_penalty
        return reward

    def compute_done(self, observation):
        pos, orn = np.split(observation, [3])
        done = pos[2] < 0.3
        return done

    def close(self):
        p.disconnect()

def test_rl_agent(model_path, num_steps=1000):
    env = Solo12Env(render=True)
    model = PPO.load(model_path)
    obs = env.reset()

    # Print the shapes for debugging
    print("Model expected observation shape:", model.policy.observation_space.shape)
    print("Environment observation shape:", obs.shape)

    positions = []
    orientations = []
    torques = []

    for _ in range(num_steps):
        action, _states = model.predict(obs)
        obs, rewards, done, truncated, info = env.step(action)

        pos, orn = p.getBasePositionAndOrientation(env.body_id)
        joint_states = p.getJointStates(env.body_id, range(p.getNumJoints(env.body_id)))
        joint_torques = [state[3] for state in joint_states]

        positions.append(pos)
        orientations.append(p.getEulerFromQuaternion(orn))
        torques.append(joint_torques)

        if done:
            obs = env.reset()

    env.close()

    positions = np.array(positions)
    orientations = np.array(orientations)
    torques = np.array(torques)

    return positions, orientations, torques

def plot_results(positions, orientations, torques, dt):
    time_steps = np.arange(len(positions)) * dt

    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    axs[0].plot(time_steps, positions[:, 0], label='X')
    axs[0].plot(time_steps, positions[:, 1], label='Y')
    axs[0].plot(time_steps, positions[:, 2], label='Z')
    axs[0].set_title('Position Over Time')
    axs[0].set_xlabel('Time [s]')
    axs[0].set_ylabel('Position [m]')
    axs[0].legend()

    axs[1].plot(time_steps, orientations[:, 0], label='Roll')
    axs[1].plot(time_steps, orientations[:, 1], label='Pitch')
    axs[1].plot(time_steps, orientations[:, 2], label='Yaw')
    axs[1].set_title('Orientation Over Time')
    axs[1].set_xlabel('Time [s]')
    axs[1].set_ylabel('Orientation [rad]')
    axs[1].legend()

    fig2, axs2 = plt.subplots(4, 3, figsize=(15, 10), sharex=True)
    axs2 = axs2.ravel()
    for i in range(12):
        axs2[i].plot(time_steps, torques[:, i])
        axs2[i].set_title(f'Joint {i} Torque')
        axs2[i].set_xlabel('Time [s]')
        axs2[i].set_ylabel('Torque [Nm]')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    dt = 0.005  # Timestep duration
    positions, orientations, torques = test_rl_agent('solo12_rl_model')
    plot_results(positions, orientations, torques, dt)

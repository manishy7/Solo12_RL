import numpy as np
import pybullet as p
import pybullet_data
from kinematic_model import robotKinematics
from pybullet_debugger import pybulletDebug  
from gaitPlanner import trotGait

class Solo12Env:
    def __init__(self):
        self.physicsClient = p.connect(p.GUI)
        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_id = p.loadURDF("/home/manishyadav/Downloads/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf", [0, 0, 0.48])
        self.max_force = 2
        self.action_space = 12  # Define the size of the action space
        self.observation_space = 37  # Define the size of the observation space based on the actual observation

        self.target_position = np.array([10, 0, 0])  # Example target position

    def reset(self):
        p.resetSimulation()
        self.robot_id = p.loadURDF("/home/manishyadav/Downloads/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf", [0, 0, 0.48])
        self.kinematics = robotKinematics()
        self.debugger = pybulletDebug()
        self.trot = trotGait()
        self.bodytoFeet0 = np.matrix([[0.1946/2, -0.15/2, 0.1],
                                      [0.1946/2, 0.15/2, 0.1],
                                      [-0.1946/2, -0.15/2, 0.1],
                                      [-0.1946/2, 0.15/2, 0.1]])
        self.steps = 0
        return self._get_observation()

    def step(self, action):
        pos, orn, L, angle, Lrot, T, sda, offset = self.debugger.cam_and_robotstates(self.robot_id)
        bodytoFeet = self.trot.loop(L, angle, Lrot, T, offset, self.bodytoFeet0, sda)
        self._apply_action(action)
        p.stepSimulation()
        obs = self._get_observation()
        reward = self._get_reward()
        done = self._is_done()
        self.steps += 1
        return obs, reward, done, {}

    def _apply_action(self, action):
        # Apply the action to the robot joints
        for i in range(len(action)):
            p.setJointMotorControl2(self.robot_id, i, p.POSITION_CONTROL, targetPosition=action[i], force=self.max_force)

    def _get_observation(self):
        # Get the current state of the robot
        pos, orn = p.getBasePositionAndOrientation(self.robot_id)
        linear_vel, angular_vel = p.getBaseVelocity(self.robot_id)
        joint_states = p.getJointStates(self.robot_id, range(self.action_space))

        joint_positions = [state[0] for state in joint_states]
        joint_velocities = [state[1] for state in joint_states]

        obs = np.concatenate([pos, orn, linear_vel, angular_vel, joint_positions, joint_velocities])
        return obs

    def _get_reward(self):
        # Reward based on distance to target and maintaining balance
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance_to_target = np.linalg.norm(self.target_position - np.array(pos))
        reward = -distance_to_target

        # Bonus for being upright
        orientation = p.getEulerFromQuaternion(p.getBasePositionAndOrientation(self.robot_id)[1])
        if abs(orientation[0]) < 0.1 and abs(orientation[1]) < 0.1:  # Roll and pitch should be small
            reward += 1.0

        # Penalize for falling over
        if pos[2] < 0.2:  # Assuming the robot is on the ground if z < 0.2
            reward -= 10.0

        return reward

    def _is_done(self):
        # Check if the episode should terminate
        pos, _ = p.getBasePositionAndOrientation(self.robot_id)
        distance_to_target = np.linalg.norm(self.target_position - np.array(pos))
        if distance_to_target < 0.5:  # Close to the target
            return True
        if self.steps >= 1000:  # Max steps per episode
            return True
        if pos[2] < 0.2:  # Robot has fallen
            return True
        return False

    def render(self, mode='human'):
        pass

    def close(self):
        p.disconnect()
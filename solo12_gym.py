import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
from kinematic_model_gym import robotKinematicsGym
from pybullet_debugger import pybulletDebug
from gaitPlanner import trotGait

class Solo12Env(gym.Env):
    def __init__(self, render=False):
        super(Solo12Env, self).__init__()

        self.dT = 0.005
        self.body_pos = [0, 0, 0.25]
        self.fixed = False
        self.render = render
        self.physics_client = None

        self.bodyId, self.jointIds = self.robot_init(self.dT, self.body_pos, self.fixed)
        self.robotKinematics = robotKinematicsGym()
        self.trot = trotGait()

        # Action and observation space
        self.action_space = spaces.Box(low=np.array([-1] * 12), high=np.array([1] * 12), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12,), dtype=np.float32)

        self.bodytoFeet0 = np.matrix([[0.0973, -0.075, 0.10], [0.0973, 0.075, 0.10], [-0.0973, -0.075, 0.10], [-0.0973, 0.075, 0.10]])

    def robot_init(self, dt, body_pos, fixed=False):
        if self.physics_client is None:
            self.physics_client = p.connect(p.GUI if self.render else p.DIRECT)
        else:
            p.resetSimulation()

        p.setGravity(0, 0, -9.8)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(
            fixedTimeStep=dt,
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
        body_id = p.loadURDF("/home/manishyadav/Downloads/opt/openrobots/share/example-robot-data/robots/solo_description/robots/solo12.urdf", body_pos, useFixedBase=fixed)
        joint_ids = []
        maxVel = 3.703
        for j in range(p.getNumJoints(body_id)):
            p.changeDynamics(body_id, j, lateralFriction=1e-5, linearDamping=0, angularDamping=0)
            p.changeDynamics(body_id, j, maxJointVelocity=maxVel)
            joint_ids.append(p.getJointInfo(body_id, j))
        return body_id, joint_ids

    def step(self, action):
        print(f"Action: {action}")  # Debugging output
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Convert action to joint angles
        fr_angles, fl_angles, br_angles, bl_angles = action[:3], action[3:6], action[6:9], action[9:12]

        print(f"Applying joint angles: FR={fr_angles}, FL={fl_angles}, BR={br_angles}, BL={bl_angles}")  # Debugging output

        for i in range(3):
            p.setJointMotorControl2(self.bodyId, i, p.POSITION_CONTROL, targetPosition=fr_angles[i], force=1.0)
            p.setJointMotorControl2(self.bodyId, 4 + i, p.POSITION_CONTROL, targetPosition=fl_angles[i], force=1.0)
            p.setJointMotorControl2(self.bodyId, 8 + i, p.POSITION_CONTROL, targetPosition=br_angles[i], force=1.0)
            p.setJointMotorControl2(self.bodyId, 12 + i, p.POSITION_CONTROL, targetPosition=bl_angles[i], force=1.0)

        p.stepSimulation()

        observation = self._get_observation()
        reward = self._compute_reward(observation)
        done = self._is_done(observation)
        info = {}

        return observation, reward, done, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        self.bodyId, self.jointIds = self.robot_init(self.dT, self.body_pos, self.fixed)
        self.set_initial_pose()
        return self._get_observation(), {}

    def render(self, mode='human'):
        if self.render:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)

    def _get_robot_state(self):
        pos, orn = p.getBasePositionAndOrientation(self.bodyId)
        return pos, orn, 0, 0, 0, 0

    def _get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.bodyId)
        pos = np.array(pos, dtype=np.float32)
        orn = np.array(orn, dtype=np.float32)

        joint_states = p.getJointStates(self.bodyId, range(p.getNumJoints(self.bodyId)))
        joint_angles = np.array([state[0] for state in joint_states[:12]], dtype=np.float32)  # Only take first 12 joint angles
        observation = joint_angles.astype(np.float32)  # Adjust to match expected shape (12,)
        return observation

    def _compute_reward(self, observation):
        pos, orn = p.getBasePositionAndOrientation(self.bodyId)
        pos_reward = -np.linalg.norm(np.array(pos[:2]))  # Reward for staying close to origin
        return pos_reward

    def _is_done(self, observation):
        pos, orn = p.getBasePositionAndOrientation(self.bodyId)
        # Check if the robot has fallen over by inspecting the orientation
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        if abs(roll) > np.pi / 4 or abs(pitch) > np.pi / 4:
            return True
        return False

    def set_initial_pose(self):
        initial_angles = [0.0, 0.6, -1.2] * 4  # Slightly bent legs for stability
        for i in range(3):
            p.setJointMotorControl2(self.bodyId, i, p.POSITION_CONTROL, targetPosition=initial_angles[i], force=1.0)
            p.setJointMotorControl2(self.bodyId, 4 + i, p.POSITION_CONTROL, targetPosition=initial_angles[3 + i], force=1.0)
            p.setJointMotorControl2(self.bodyId, 8 + i, p.POSITION_CONTROL, targetPosition=initial_angles[6 + i], force=1.0)
            p.setJointMotorControl2(self.bodyId, 12 + i, p.POSITION_CONTROL, targetPosition=initial_angles[9 + i], force=1.0)
        for _ in range(100):
            p.stepSimulation()

    def robot_stepsim(self, body_id, body_pos, body_orn, body2feet):
        fr_index, fl_index, br_index, bl_index = 3, 7, 11, 15
        maxForce = 1.0  # Adjusted force for stability

        body_orn = np.array(body_orn, dtype=np.float32)
        body_pos = np.array(body_pos, dtype=np.float32)

        fr_angles, fl_angles, br_angles, bl_angles, body2feet_ = self.robotKinematics.solve(body_orn, body_pos, body2feet)
        for i in range(3):
            p.setJointMotorControl2(body_id, i, p.POSITION_CONTROL, targetPosition=fr_angles[i], force=maxForce)
            p.setJointMotorControl2(body_id, 4 + i, p.POSITION_CONTROL, targetPosition=fl_angles[i], force=maxForce)
            p.setJointMotorControl2(body_id, 8 + i, p.POSITION_CONTROL, targetPosition=br_angles[i], force=maxForce)
            p.setJointMotorControl2(body_id, 12 + i, p.POSITION_CONTROL, targetPosition=bl_angles[i], force=maxForce)
        p.stepSimulation()

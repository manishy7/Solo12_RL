import pybullet as p
import time
import numpy as np
import sys

class pybulletDebug:
    def __init__(self):
        #Camera paramers to be able to yaw pitch and zoom the camera (Focus remains on the robot) 
        self.cyaw=90
        self.cpitch=-7
        self.cdist=0.30
        time.sleep(0.5)
        
        self.xId = p.addUserDebugParameter("x" , -0.02 , 0.02 , 0.)
        self.yId = p.addUserDebugParameter("y" , -0.02 , 0.02 , 0.)
        self.zId = p.addUserDebugParameter("z" , -0.02 , 0.02 , 0.)
        self.rollId = p.addUserDebugParameter("roll" , -np.pi/4 , np.pi/4 , 0.)
        self.pitchId = p.addUserDebugParameter("pitch" , -np.pi/4 , np.pi/4 , 0.)
        self.yawId = p.addUserDebugParameter("yaw" , -np.pi/4 , np.pi/4 , 0.)
        self.LId = p.addUserDebugParameter("L" , -0.50 , 1 , 0.)
        self.LrotId = p.addUserDebugParameter("Lrot" , -1.50 , 1.50 , 0.)
        self.angleId = p.addUserDebugParameter("angleWalk" , -180. , 180. , 0.)
        self.periodId = p.addUserDebugParameter("stepPeriod" , 0.1 , 3. , 1.0)
        self.step_dur_asym = p.addUserDebugParameter("step_dur_asym" , -2 , 2. , 0.0)
        self.trotId = p.addUserDebugParameter("TROT" , 1 , 0 , 1)
        self.boundId = p.addUserDebugParameter("BOUND" , 1 , 0 , 1)
        
    
    def cam_and_robotstates(self , boxId):
        #orientation of camara
        cubePos, cubeOrn = p.getBasePositionAndOrientation(boxId)
        p.resetDebugVisualizerCamera( cameraDistance=self.cdist, cameraYaw=self.cyaw, cameraPitch=self.cpitch, cameraTargetPosition=cubePos)
        keys = p.getKeyboardEvents()
        #Keys to change camera
        if keys.get(104):  #H
            self.cyaw+=1
        if keys.get(102):  #F
            self.cyaw-=1
        if keys.get(116):  #T
            self.cpitch+=1
        if keys.get(103):  #G
            self.cpitch-=1
        if keys.get(122):  #Z
            self.cdist+=0.01
        if keys.get(120):  #X
            self.cdist-=0.01
        if keys.get(27):  #ESC
            p.disconnect()
            time.sleep(2)
            
        #   sys.exit()
        #read position from debug
        pos = np.array([p.readUserDebugParameter(self.xId),p.readUserDebugParameter(self.yId), p.readUserDebugParameter(self.zId)])
        orn = np.array([p.readUserDebugParameter(self.rollId),p.readUserDebugParameter(self.pitchId), p.readUserDebugParameter(self.yawId)])
        L = p.readUserDebugParameter(self.LId)
        Lrot = p.readUserDebugParameter(self.LrotId)
        angle = p.readUserDebugParameter(self.angleId)
        T = p.readUserDebugParameter(self.periodId)
        trot=p.readUserDebugParameter(self.trotId)
        bound=p.readUserDebugParameter(self.boundId)
        
        if trot==1:
          offset=[0.5, 0., 0., 0.5]
        elif bound==1:
          offset=[0.5, 0.5, 0., 0.]
        else:
          offset=[0.5, 0., 0., 0.5]
        
        
        return pos , orn , L , angle , Lrot , T , p.readUserDebugParameter(self.step_dur_asym), offset

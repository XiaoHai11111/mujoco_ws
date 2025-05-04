"""
    File Name: trajector_plan_toppra.py
    Author: XiaoHai
    Date: 2025-05-03
    Version: 1.0
    Description:加载dummy的urdf描述文件,使用pinocchio库进行运动学逆解,
        控制机械臂沿着z轴运动
    Usage:通过以下指令运行代码
        $ conda activate mujoco_py
        $ python trajector_plan_toppra.py
"""

from scipy.optimize import minimize
import numpy as np
from numpy.linalg import norm, solve
import toppra as ta
import toppra.constraint as constraint
import toppra.algorithm as algo
import pinocchio
import time
import mujoco_viewer

class Test(mujoco_viewer.CustomViewer):
    def __init__(self, path):
        super().__init__(path, 1.5, azimuth=135, elevation=-30)
        self.path = path
        self.runBefor()
    
    def runBefor(self):
        robot = pinocchio.buildModelFromUrdf('../dummy_description/urdf/dummy_gripper.urdf')
        print('robot name: ' + robot.name)
        
        # 关节角度可以通过mujoco中滑块控制并通过ctrl+c复制获得
        # <key qpos='0.00177052 -0.000159299 0.000142973 3.72164e-07 6.41045e-05 1.70419e-08 4.17917e-10 4.29006e-10'/>
        # <key qpos='0.982613 -0.000582215 0.000143164 3.7986e-07 6.40952e-05 3.17941e-08 1.68279e-08 -1.75817e-08'/>
        # <key qpos='-0.691546 0.20982 -0.0653241 -0.0314229 0.735041 2.47132e-08 0.410135 0.410139'/>
        # <key qpos='-0.359438 0.21035 0.594998 -0.0314192 0.735002 8.99848e-09 0.00320611 0.00322535'/>
        # <key qpos='-1.44034 0.207633 0.59502 -0.0314192 0.735002 9.26483e-09 0.777792 0.777779'/>
        # <key qpos='0.00177052 -0.000159299 0.000142973 3.72164e-07 6.41045e-05 1.70419e-08 4.17917e-10 4.29006e-10'/>
        
        way_pts = [
            [0.00177052, -0.000159299, 0.000142973, 3.72164e-07, 6.41045e-05, 1.70419e-08, 4.17917e-10, 4.29006e-10],
            [0.982613, -0.000582215, 0.000143164, 3.7986e-07, 6.40952e-05, 3.17941e-08, 1.68279e-08, -1.75817e-08],
            [-0.691546, 0.20982, -0.0653241, -0.0314229, 0.735041, 2.47132e-08, 0.410135, 0.410139],
            [-0.359438, 0.21035, 0.594998, -0.0314192, 0.735002, 8.99848e-09, 0.00320611, 0.00322535],
            [-1.44034, 0.207633, 0.59502, -0.0314192, 0.735002, 9.26483e-09, 0.777792, 0.777779],
            [0.00177052, -0.000159299, 0.000142973, 3.72164e-07, 6.41045e-05, 1.70419e-08, 4.17917e-10, 4.29006e-10]
            ]
           
        path_scalars = np.linspace(0, 1, len(way_pts))
        path = ta.SplineInterpolator(path_scalars, way_pts)
        vlim = np.vstack([-robot.velocityLimit, robot.velocityLimit]).T
        al = np.array([2,] * robot.nv)
        alim = np.vstack([-al, al]).T
        pc_vel = constraint.JointVelocityConstraint(vlim)
        pc_acc = constraint.JointAccelerationConstraint(
            alim, discretization_scheme=constraint.DiscretizationType.Interpolation)
        
        instance = ta.algorithm.TOPPRA([pc_vel, pc_acc],path,solver_wrapper="seidel")
        jnt_traj = instance.compute_trajectory(0, 0)
        ts_sample = np.linspace(0, jnt_traj.get_duration(), 1000)
        self.qs_sample = jnt_traj.eval(ts_sample)
        self.index = 0
    
    def runFunc(self):
        if self.index < len(self.qs_sample):
            self.data.qpos[:8] = self.qs_sample[self.index][:8]
            self.index += 1
        else:
            self.data.qpos[:8] = self.qs_sample[-1][:8]
        time.sleep(0.01)

test = Test("../dummy_description/mjcf/scene.xml")
test.run_loop()
"""
    File Name: test_pinpcchio.py
    Author: XiaoHai
    Date: 2025-05-03
    Version: 1.0
    Description:加载dummy的urdf描述文件,使用pinocchio库进行运动学正解,
        并打印每个关节的姿态
    Usage:通过以下指令运行代码
        $ conda activate mujoco_py
        $ python test_pinpcchio.py
"""
from pathlib import Path
from sys import argv
 
import pinocchio
 
# Load the urdf model
model = pinocchio.buildModelFromUrdf("../dummy_description/urdf/dummy_gripper.urdf")
print("model name: " + model.name)
 
# Create data required by the algorithms
data = model.createData()
 
# Sample a random configuration在角度范围内随机生成角度求解
q = pinocchio.randomConfiguration(model)
print(f"q: {q.T}")
 
# Perform the forward kinematics over the kinematic tree
pinocchio.forwardKinematics(model, data, q)
 
# Print out the placement of each joint of the kinematic tree
for name, oMi in zip(model.names, data.oMi):
    print("{:<24} : {: .2f} {: .2f} {: .2f}".format(name, *oMi.translation.T.flat))
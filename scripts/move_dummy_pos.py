"""
    File Name: move_dummy_pos.py
    Author: XiaoHai
    Date: 2025-05-03
    Version: 1.0
    Description:通过按键改变机械臂的末端位置,并使用pinocchio库进行运动学逆解，
        通过逆解控制机械臂在关节空间运动
    Usage:通过以下指令运行代码
        $ conda activate mujoco_py
        $ python get_body_pos.py
"""
import mujoco
import numpy as np
import glfw
import mujoco.viewer
from pynput import keyboard
import pinocchio
from numpy.linalg import norm, solve

def inverse_kinematics(current_q, target_dir, target_pos):
    urdf_filename = '../dummy_description/urdf/dummy_gripper.urdf'
    model = pinocchio.buildModelFromUrdf(urdf_filename)
    data = model.createData()

    JOINT_ID = 6
    oMdes = pinocchio.SE3(target_dir, np.array(target_pos))

    q = current_q
    eps = 1e-3
    IT_MAX = 1000
    DT = 1e-2
    damp = 1e-12

    i = 0
    while True:
        pinocchio.forwardKinematics(model, data, q)
        iMd = data.oMi[JOINT_ID].actInv(oMdes)
        err = pinocchio.log(iMd).vector

        if norm(err) < eps:
            success = True
            break
        if i >= IT_MAX:
            success = False
            break

        J = pinocchio.computeJointJacobian(model, data, q, JOINT_ID)
        J = -np.dot(pinocchio.Jlog6(iMd.inverse()), J)
        v = -J.T.dot(solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pinocchio.integrate(model, q, v * DT)

        i += 1

    if success:
        print("Convergence achieved!")
    else:
        print("Warning: the iterative algorithm has not reached convergence to the desired precision")

    print(f"result: {q.flatten().tolist()}")
    print(f"final error: {err.T}")
    return q.flatten().tolist()

key_states = {
    keyboard.Key.up: False,
    keyboard.Key.down: False,
    keyboard.Key.left: False,
    keyboard.Key.right: False,
    keyboard.Key.alt_l: False,
    keyboard.Key.alt_r: False,
}

def on_press(key):
    if key in key_states:
        key_states[key] = True

def on_release(key):
    if key in key_states:
        key_states[key] = False

listener = keyboard.Listener(on_press=on_press, on_release=on_release)
listener.start()

model = mujoco.MjModel.from_xml_path('../dummy_description/mjcf/scene.xml')
data = mujoco.MjData(model)

class CustomViewer:
    def __init__(self, model, data):
        self.handle = mujoco.viewer.launch_passive(model, data)
        self.pos_step = 0.001  # 定义每次按键移动的步长

        self.end_effector_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'wrist3_link')
        print(f"End effector ID: {self.end_effector_id}")
        if self.end_effector_id == -1:
            print("Warning: Could not find the end effector with the given name.")

        self.initial_q = data.qpos[:model.nq].copy()
        print(f"Initial joint positions: {self.initial_q}")
        theta = -np.pi / 2
        self.R_x = np.array([
            [np.cos(theta), 0, -np.sin(theta)],
            [0, 1, 0],
            [np.sin(theta), 0, np.cos(theta)]
        ])

        self.target_pos = data.body(self.end_effector_id).xpos.copy()
        self.new_q = self.initial_q

    def is_running(self):
        return self.handle.is_running()

    def sync(self):
        self.handle.sync()

    @property
    def cam(self):
        return self.handle.cam

    @property
    def viewport(self):
        return self.handle.viewport

    def run_loop(self):
        while self.is_running():
            # 获取当前末端执行器的位置
            current_pos = data.body(self.end_effector_id).xpos

            # 根据按键状态调整目标位置
            if key_states[keyboard.Key.up]:
                self.target_pos[2] += self.pos_step
            if key_states[keyboard.Key.down]:
                self.target_pos[2] -= self.pos_step
            if key_states[keyboard.Key.left]:
                self.target_pos[1] -= self.pos_step
            if key_states[keyboard.Key.right]:
                self.target_pos[1] += self.pos_step
            if key_states[keyboard.Key.alt_l]:
                self.target_pos[0] -= self.pos_step
            if key_states[keyboard.Key.alt_r]:
                self.target_pos[0] += self.pos_step

            # 调用逆运动学函数计算新的关节角度
            self.new_q = inverse_kinematics(self.initial_q, self.R_x, self.target_pos)
            data.qpos[:model.nq] = self.new_q

            # 更新仿真状态
            mujoco.mj_step(model, data)
            self.sync()

viewer = CustomViewer(model, data)
viewer.cam.distance = 1.5
viewer.cam.azimuth = 135
viewer.cam.elevation = -30
viewer.run_loop()
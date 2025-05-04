# dummy在mujoco中仿真

## 末端位姿实时显示

通过mjcf描述文件中的body name查找连杆并控制，主要的函数

```python
self.end_effector_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'wrist3_link')
```

## 关节空间的控制

主要的函数

```python
initial_q = data.qpos[:7].copy()

data.qpos[:7] = new_q

mujoco.mjv_updateScene(model, data, opt, pert, cam, mujoco.mjtCatBit.mjCAT_ALL.value, scene)
```

## pinocchio库使用

pinocchio动力学库c++安装[教程](https://www.bilibili.com/opus/943181170510659585)

pinocchio动力学python安装：

```
pip install pin
```

具体参考代码：dummy_inverse_dynamics.cpp和test_pinpcchio.py

## 通过pinocchio实现机械臂运动学逆解

思路：获取机械臂的初始轨迹，使机械臂沿着Z轴向下运动，每次在当前姿态Z轴自增并通过pinocchio进行逆解，控制机械臂关节转动

参考代码：ctrl_ee_pin.py



## 通过pinocchio实现机械臂逆动力学

参考代码：dummy_inverse_dynamics.cpp

## 通过TOPPRA实现机械臂轨迹规划

toppra代码[仓库](https://github.com/hungpham2511/toppra)

可通过pip install toppra直接安装


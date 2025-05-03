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


from Dragon import Dragon
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import threading

dragon = Dragon()
dragon.reset_joint_pos("joint1_pitch",-1.5)
dragon.reset_joint_pos("joint2_pitch", 1.5)
dragon.reset_joint_pos("joint3_pitch", 1.5)
dragon.hover()
dragon.step()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
dragon.plot_on_ax(ax)

real_W_hist = []
pred_W_hist = []

u = 0.1
for _ in range(10):
    real_W = dragon.wrench()
    real_W_hist.append(real_W)

    thrusts = []
    for i in range(dragon.num_modules):
      phi = dragon.get_joint_pos(f"G{i+1}")
      theta = dragon.get_joint_pos(f"F{i+1}")
      thrust = dragon.module_thrust(i+1)

      phi += u
      theta += u
      thrust += u

      dragon.reset_joint_pos(f"G{i+1}", phi)
      dragon.reset_joint_pos(f"F{i+1}", theta)

      thrusts.append(thrust / 2)
      thrusts.append(thrust / 2)

    dragon.step()
    dragon.thrust(thrusts)

dragon = Dragon()
dragon.hover()
dragon.step()

phi = np.zeros(dragon.num_modules)
theta = np.zeros(dragon.num_modules)
thrust = np.zeros(dragon.num_modules)

for _ in range(10):
  wrenches = []


  thrusts = []
  for i in range(dragon.num_modules):
    phi = dragon.get_joint_pos(f"G{i+1}")
    theta = dragon.get_joint_pos(f"F{i+1}")
    thrust = dragon.module_thrust(i+1)

    W = dragon.module_wrench(i+1)
    wrenches.append(W)

    phi += u
    theta += u
    thrust += u

    dragon.reset_joint_pos(f"G{i+1}", phi)
    dragon.reset_joint_pos(f"F{i+1}", theta)

    thrusts.append(thrust / 2)
    thrusts.append(thrust / 2)

  pred_W_hist.append(np.sum(wrenches, axis=0))
  dragon.step()
  dragon.thrust(thrusts)


fig = plt.figure(figsize=(10, 5))
names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
for i in range(6):
  fig.add_subplot(2, 3, i + 1)
  plt.plot([w[i] for w in real_W_hist], label="Real Wrench", color='blue')
  plt.plot([w[i] for w in pred_W_hist], label="Predicted Wrench", color='orange')
  plt.title(names[i])
  plt.xlabel("Time step")

plt.legend()
plt.tight_layout()
plt.show()

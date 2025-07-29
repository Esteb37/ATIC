from sim.Dragon import Dragon
import numpy as np
import matplotlib.pyplot as plt
import pybullet as p
import threading

real_dragon = Dragon()
real_dragon.reset_joint_pos("joint1_pitch",-1.5)
real_dragon.reset_joint_pos("joint2_pitch", 1.5)
real_dragon.reset_joint_pos("joint3_pitch", 1.5)

real_dragon.reset_joint_pos("F1", 1)
real_dragon.reset_joint_pos("F2", 1)
real_dragon.reset_joint_pos("F3", -1)
real_dragon.reset_joint_pos("F4", -1)


real_dragon.hover()
real_dragon.step()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
real_dragon.plot_on_ax(ax)

real_W_hist = []
real_cog_hist = []
pred_W_hist = []
pred_cog_hist = []
calc_W_hist = []

u = 0.5

for _ in range(10):
    real_W = real_dragon.wrench()
    real_W_hist.append(real_W.copy())

    thrusts = []
    calc_Ws = []
    for i in range(real_dragon.num_modules):
      phi = real_dragon.get_joint_pos(f"G{i+1}")
      theta = real_dragon.get_joint_pos(f"F{i+1}")
      thrust = real_dragon.module_thrust(i+1)

      calc_W, _ = real_dragon.linearize_module(i+1, phi, theta, thrust)
      calc_Ws.append(calc_W)

      phi += u
      theta += u
      thrust += u

      real_dragon.reset_joint_pos(f"G{i+1}", phi)
      real_dragon.reset_joint_pos(f"F{i+1}", theta)

      thrusts.append(thrust / 2)
      thrusts.append(thrust / 2)

    calc_W_hist.append(np.sum(calc_Ws, axis=0))
    real_dragon.step()
    real_dragon.thrust(thrusts)
    real_cog_hist.append(real_dragon.center_of_gravity.copy())

np.save("cog_hist.npy", np.array(real_cog_hist))

p.resetSimulation()
pred_dragon = Dragon()
pred_dragon.reset_joint_pos("joint1_pitch",-1.5)
pred_dragon.reset_joint_pos("joint2_pitch", 1.5)
pred_dragon.reset_joint_pos("joint3_pitch", 1.5)

pred_dragon.reset_joint_pos("F1", 1)
pred_dragon.reset_joint_pos("F2", 1)
pred_dragon.reset_joint_pos("F3", -1)
pred_dragon.reset_joint_pos("F4", -1)

pred_dragon.hover()
pred_dragon.step()

fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
pred_dragon.plot_on_ax(ax)

phi = np.zeros(pred_dragon.num_modules)
theta = np.zeros(pred_dragon.num_modules)
thrust = np.zeros(pred_dragon.num_modules)

cog_hist = np.load("cog_hist.npy")
for k in range(10):
  wrenches = []
  thrusts = []
  for i in range(pred_dragon.num_modules):

    if k == 0:
      phi[i] = pred_dragon.module_phi(i+1)
      theta[i] = pred_dragon.module_theta(i+1)
      thrust[i] = pred_dragon.module_thrust(i+1)

    W, A = pred_dragon.linearize_module(i+1, phi[i], theta[i], thrust[i])

    wrenches.append(W)

    phi[i] += u
    theta[i] += u
    thrust[i] += u

  pred_dragon.center_of_gravity = cog_hist[k]
  pred_W_hist.append(np.sum(wrenches, axis=0))
  pred_cog_hist.append(pred_dragon.center_of_gravity.copy())


fig = plt.figure(figsize=(10, 5))
names = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
for i in range(6):
  fig.add_subplot(3, 3, i + 1)
  plt.plot([w[i] for w in real_W_hist], label="Real Wrench", color='blue')
  plt.plot([w[i] for w in pred_W_hist], label="Predicted Wrench", color='orange')
  # plt.plot([w[i] for w in calc_W_hist], label="Calculated Wrench", color='green')
  plt.title(names[i])
  plt.xlabel("Time step")

for i in range(3):
  fig.add_subplot(3, 3, i + 7)
  plt.plot([cog[i] for cog in real_cog_hist], label="Real COG", color='blue')
  plt.plot([cog[i] for cog in pred_cog_hist], label="Predicted COG", color='orange')
  plt.title(f"COG {i}")
  plt.xlabel("Time step")

plt.legend()
plt.tight_layout()
plt.show()

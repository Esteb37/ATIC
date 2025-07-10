from Dragon import Dragon
import matplotlib.pyplot as plt
import numpy as np
import pybullet as p

dragon = Dragon()

dragon.reset_joint_pos("joint1_pitch",-1)
dragon.reset_joint_pos("joint2_yaw", 1)
dragon.reset_joint_pos("joint3_pitch",-1)
dragon.reset_joint_pos("joint2_pitch", -1)
dragon.step()
dragon.hover()

A = np.array([
                [1/2, 1/2, 0,   0  ],
                [1/3, 1/3, 1/3, 0  ],
                [0,   1/3, 1/3, 1/3],
                [0,   0,   1/3, 2/3]
            ])

w = np.ones(dragon.num_modules)

b = np.array([0, 0, 1])

lamb = np.array([dragon.module_thrust(1),
                 dragon.module_thrust(2),
                 dragon.module_thrust(3),
                 dragon.module_thrust(4)])

u = np.array([p.rotateVector(dragon.link_orientation("F1"), b),
              p.rotateVector(dragon.link_orientation("F2"), b),
              p.rotateVector(dragon.link_orientation("F3"), b),
              p.rotateVector(dragon.link_orientation("F4"), b)])

F = [lamb[i] * u[i] for i in range(dragon.num_modules)]

F_real = dragon.sum_of_forces()
history = []
for i in range(20):
  F = A @ F
  history.append(F.copy() * 4)

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')


ax2 = fig.add_subplot(222)
x = np.arange(len(history))
ax2.plot(x, [h[0][0] for h in history], label='CoG1 X', color='blue')
ax2.plot(x, [h[1][0] for h in history], label='CoG2 X', color='green')
ax2.plot(x, [h[2][0] for h in history], label='CoG3 X', color='red')
ax2.plot(x, [h[3][0] for h in history], label='CoG4 X', color='orange')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Fx magnitude')
ax2.axhline(F_real[0], color='black', linestyle='--', label='Real CoG X')
ax2.legend()
ax3 = fig.add_subplot(223)
ax3.plot(x, [h[0][1] for h in history], label='CoG1 Y', color='blue')
ax3.plot(x, [h[1][1] for h in history], label='CoG2 Y', color='green')
ax3.plot(x, [h[2][1] for h in history], label='CoG3 Y', color='red')
ax3.plot(x, [h[3][1] for h in history], label='CoG4 Y', color='orange')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Fy magnitude')
ax3.axhline(F_real[1], color='black', linestyle='--', label='Real CoG Y')
ax3.legend()
ax4 = fig.add_subplot(224)
ax4.plot(x, [h[0][2] for h in history], label='CoG1 Z', color='blue')
ax4.plot(x, [h[1][2] for h in history], label='CoG2 Z', color='green')
ax4.plot(x, [h[2][2] for h in history], label='CoG3 Z', color='red')
ax4.plot(x, [h[3][2] for h in history], label='CoG4 Z', color='orange')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Fz magnitude')
ax4.axhline(F_real[2], color='black', linestyle='--', label='Real CoG Z')
ax4.legend()

F_real /= 9.81
history = [[f / 9.81  for f in h] for h in history]

cog = dragon.center_of_gravity
# plot history
for i in range(len(history)):
    ax.quiver(cog[0], cog[1], cog[2], history[i][0][0], history[i][0][1], history[i][0][2], color='blue', alpha=0.5, label ="F1" if i == 0 else None)
    ax.quiver(cog[0], cog[1], cog[2], history[i][1][0], history[i][1][1], history[i][1][2], color='green', alpha=0.5, label ="F2" if i == 0 else None)
    ax.quiver(cog[0], cog[1], cog[2], history[i][2][0], history[i][2][1], history[i][2][2], color='red', alpha=0.5, label ="F3" if i == 0 else None)
    ax.quiver(cog[0], cog[1], cog[2], history[i][3][0], history[i][3][1], history[i][3][2], color='orange', alpha=0.5, label ="F4" if i == 0 else None)

ax.quiver(cog[0], cog[1], cog[2], F_real[0], F_real[1], F_real[2], color='black', label='Real Force')

ax.legend()
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-1, 1])
ax.set_zlim([2, 4])

dragon.plot_on_ax(ax, CoG=False, forces=True)
plt.show()

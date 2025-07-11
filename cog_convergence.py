from Dragon import Dragon
import matplotlib.pyplot as plt
import numpy as np

dragon = Dragon()

dragon.reset_joint_pos("joint1_pitch",-1)
dragon.reset_joint_pos("joint2_yaw", 1)
dragon.reset_joint_pos("joint3_pitch",-1)
dragon.reset_joint_pos("joint2_pitch", -1)
dragon.step()
dragon.hover()

A = np.array([
                [2/3, 1/3, 0,   0  ],
                [1/3, 1/3, 1/3, 0  ],
                [0,   1/3, 1/3, 1/3],
                [0,   0,   1/3, 2/3]
            ])

G = np.array([dragon.module_cog(1),
              dragon.module_cog(2),
              dragon.module_cog(3),
              dragon.module_cog(4)])

CoG_real = dragon.center_of_gravity

CoG = G.copy()
w = np.ones(dragon.num_modules)

history = []
for i in range(20):
  G = A @ G
  CoG = G.copy()
  history.append(CoG.copy())

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(221, projection='3d')
ax.set_xlim([-0.5, 1.5])
ax.set_ylim([-1, 1])
ax.set_zlim([2, 4])

# plot history
for i in range(len(history)):
    ax.scatter(history[i][0][0], history[i][0][1], history[i][0][2], color='blue', alpha=0.5, label ="CoG1" if i == 0 else None)
    ax.scatter(history[i][1][0], history[i][1][1], history[i][1][2], color='green', alpha=0.5, label ="CoG2" if i == 0 else None)
    ax.scatter(history[i][2][0], history[i][2][1], history[i][2][2], color='red', alpha=0.5, label ="CoG3" if i == 0 else None)
    ax.scatter(history[i][3][0], history[i][3][1], history[i][3][2], color='orange', alpha=0.5, label ="CoG4" if i == 0 else None)

ax.legend()

ax2 = fig.add_subplot(222)
# Convergence of X
x = np.arange(len(history))
ax2.plot(x, [h[0][0] for h in history], label='CoG1 X', color='blue')
ax2.plot(x, [h[1][0] for h in history], label='CoG2 X', color='green')
ax2.plot(x, [h[2][0] for h in history], label='CoG3 X', color='red')
ax2.plot(x, [h[3][0] for h in history], label='CoG4 X', color='orange')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('X Position')
ax2.set_ylim([0, 1])
ax2.axhline(CoG_real[0], color='black', linestyle='--', label='Real CoG X')
ax2.legend()
ax3 = fig.add_subplot(223)
# Convergence of Y
ax3.plot(x, [h[0][1] for h in history], label='CoG1 Y', color='blue')
ax3.plot(x, [h[1][1] for h in history], label='CoG2 Y', color='green')
ax3.plot(x, [h[2][1] for h in history], label='CoG3 Y', color='red')
ax3.plot(x, [h[3][1] for h in history], label='CoG4 Y', color='orange')
ax3.set_xlabel('Iteration')
ax3.set_ylabel('Y Position')
ax3.set_ylim([0, 1])
ax3.axhline(CoG_real[1], color='black', linestyle='--', label='Real CoG Y')
ax3.legend()
ax4 = fig.add_subplot(224)
# Convergence of Z
ax4.plot(x, [h[0][2] for h in history], label='CoG1 Z', color='blue')
ax4.plot(x, [h[1][2] for h in history], label='CoG2 Z', color='green')
ax4.plot(x, [h[2][2] for h in history], label='CoG3 Z', color='red')
ax4.plot(x, [h[3][2] for h in history], label='CoG4 Z', color='orange')
ax4.set_xlabel('Iteration')
ax4.set_ylabel('Z Position')
ax4.set_ylim([2, 4])
ax4.axhline(CoG_real[2], color='black', linestyle='--', label='Real CoG Z')
ax4.legend()

dragon.plot_on_ax(ax, CoG = True, forces = False)
plt.show()

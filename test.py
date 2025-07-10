import torch
import pytorch_kinematics as pk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

urdf = "dragon.urdf"
robot = pk.build_chain_from_urdf(open(urdf, mode="rb").read())
robot.print_tree()

dof = len(robot.get_joint_parameter_names())

# Set joint angles (example with 0s, adjust as needed)
joint_angles = torch.zeros(1, dof, dtype=torch.float32)

# Forward kinematics to get transforms of all links
fk_results = robot.forward_kinematics(joint_angles)

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

def get_color(link):
  if "rotor" in link or link[0] == "F":
    return "black"
  if link[0] == "L" or link[0] == "G":
    return "red"
  else:
    return "blue"

def plot_box(ax, center, size, color='gray', alpha=0.3, name=None):

    cx, cy, cz = center
    lx, ly, lz = size[0]/2, size[1]/2, size[2]/2

    # 8 vertices of the box
    vertices = torch.tensor([
        [cx - lx, cy - ly, cz - lz],
        [cx + lx, cy - ly, cz - lz],
        [cx + lx, cy + ly, cz - lz],
        [cx - lx, cy + ly, cz - lz],
        [cx - lx, cy - ly, cz + lz],
        [cx + lx, cy - ly, cz + lz],
        [cx + lx, cy + ly, cz + lz],
        [cx - lx, cy + ly, cz + lz],
    ])
    # Faces
    faces = [
        [vertices[j] for j in [0,1,2,3]],
        [vertices[j] for j in [4,5,6,7]],
        [vertices[j] for j in [0,1,5,4]],
        [vertices[j] for j in [2,3,7,6]],
        [vertices[j] for j in [1,2,6,5]],
        [vertices[j] for j in [4,7,3,0]],
    ]
    ax.add_collection3d(Poly3DCollection(faces, facecolors=color, linewidths=0.5, edgecolors='k', alpha=alpha))

def plot_sphere(ax, center, radius=0.05, color='blue', alpha=0.5):
    u = torch.linspace(0, 2 * torch.pi, 100)
    v = torch.linspace(0, torch.pi, 100)
    x = center[0] + radius * torch.outer(torch.cos(u), torch.sin(v))
    y = center[1] + radius * torch.outer(torch.sin(u), torch.sin(v))
    z = center[2] + radius * torch.outer(torch.ones_like(u), torch.cos(v))
    ax.plot_surface(x, y, z, color=color, alpha=alpha)

def plot_disc(ax, center, radius=0.05, color='blue', alpha=0.5):
    u = torch.linspace(0, 2 * torch.pi, 100)
    x = center[0] + radius * torch.cos(u)
    y = center[1] + radius * torch.sin(u)
    z = center[2] * torch.ones_like(u)
    ax.plot(x, y, z, color=color, alpha=alpha)

for link, transform in fk_results.items():
  if "origin" in link:
    continue
  color = get_color(link)
  m = transform.get_matrix()
  location = m[:, :3, 3][0]

  if link[0] == "G":
    plot_box(ax, location, size=[0.3, 0.05, 0.05], color=color, alpha=0.5, name = link)

  if link[0] == "F":
    plot_box(ax, location, size=[0.05, 0.30, 0.01], color=color, alpha=0.5, name = link)

  if "yaw" in link:
    plot_sphere(ax, location, radius=0.025, color=color, alpha=0.5)

  if "rotor" in link:
    plot_disc(ax, location, radius=0.05, color=color, alpha=0.5)

# Set width and height of the plot
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
ax.set_xlim([-0.2, 1.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([-0.5, 0.5])
ax.set_title("Forward Kinematics Results")

plt.legend()
plt.show()

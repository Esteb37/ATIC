import pybullet as p
import pybullet_data
import matplotlib.pyplot as plt
from time import sleep

class Dragon():

    def __init__(self, urdf_path = "dragon.urdf"):

        p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.robot_id = p.loadURDF(urdf_path, useFixedBase=True)

        p.setGravity(0, 0, -9.81)

        self.num_links = p.getNumJoints(self.robot_id)
        self.total_mass = 0.0
        self.weighted_com = [0.0, 0.0, 0.0]
        self.center_of_gravity = [0.0, 0.0, 0.0]

        self.get_com()

    def get_com(self):
        for link_index in range(-1, self.num_links):  # -1 is base
            dynamics = p.getDynamicsInfo(self.robot_id, link_index)
            mass = dynamics[0]
            if mass == 0:
                continue

            com_pos, _ = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)[0:2] if link_index != -1 \
                        else p.getBasePositionAndOrientation(self.robot_id)[0:2]

            self.weighted_com = [w + mass * c for w, c in zip(self.weighted_com, com_pos)]
            self.total_mass += mass

        self.center_of_gravity = [w / self.total_mass for w in self.weighted_com]
        return self.center_of_gravity

    def set_pos(self, name, value):
        name = name.lower()
        idx = -1
        for i in range(self.num_links):
            link_name = p.getJointInfo(self.robot_id, i)[12].decode('utf-8').lower()
            print(link_name)
            if name in link_name:
                idx = i
                break
        if idx == -1:
            raise ValueError(f"Link '{name}' not found in the robot.")
        p.setJointMotorControl2(self.robot_id, idx, p.POSITION_CONTROL, targetPosition=value)

    def step(self):
        p.stepSimulation()
        self.get_com()
        sleep(1/240)

    def _get_color(self, link):
        if "rotor" in link or link[0] == "F":
            return "black"
        if link[0] == "L" or link[0] == "G":
            return "red"
        else:
            return "blue"

    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(self.center_of_gravity[0], self.center_of_gravity[1], self.center_of_gravity[2], color='g', s=100, label='Center of Gravity')

        # plot all links
        for link_index in range(-1, self.num_links):

            link_name = p.getBodyInfo(self.robot_id)[1].decode('utf-8') if link_index == -1 else p.getJointInfo(self.robot_id, link_index)[12].decode('utf-8')

            if "origin" in link_name or "modular" in link_name:
                continue

            com_pos, _ = p.getLinkState(self.robot_id, link_index, computeForwardKinematics=True)[0:2] if link_index != -1 \
                        else p.getBasePositionAndOrientation(self.robot_id)[0:2]

            color = self._get_color(link_name)
            ax.scatter(com_pos[0], com_pos[1], com_pos[2], color=color, s=20, label=link_name)

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim([-0.2, 1.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        ax.set_title('Dragon URDF Center of Mass Visualization')
        # ax.legend()
        plt.show()


def main():
    dragon = Dragon()

    dragon.set_pos("joint2_yaw", 3.14)
    dragon.set_pos("joint3_pitch", 3.14)

    for i in range(100):
        dragon.step()

    dragon.plot()

if __name__ == "__main__":
    main()
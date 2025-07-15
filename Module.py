from Dragon import Dragon
import numpy as np
import pybullet as p

class Module:
    def __init__(self, index, dragon: Dragon):
        self.dragon = dragon
        self.index = index
        self.vec_module = "F" + str(index)

    def thrust(self):
        return self.dragon.module_thrust(self.index)

    def com(self):
        return self.dragon.module_com(self.index)

    def vec_position(self, frame):
        return self.dragon.link_position(self.vec_module) - frame

    def vec_orientation(self):
        return self.dragon.link_orientation(self.vec_module)

    def u(self):
        ez = np.array([0, 0, 1])
        return p.rotateVector(self.orientation(), ez)

    def v(self, frame):
        rel_position = self.position(frame)
        return np.cross(rel_position, self.u())

    def force(self):
        return self.thrust() * self.u()

    def torque(self, frame):
        return self.thrust() * self.v(frame)

    def wrench(self, frame):
        return np.concatenate((self.force(), self.torque(frame)))

    def q_yaw(self):
        return self.dragon.joint_position("joint" + str(self.index) + "_yaw")

    def q_pitch(self):
        return self.dragon.joint_position("joint" + str(self.index) + "_pitch")

    def theta(self):
        return self.dragon.joint_position("L" + str(self.index) + "_vec_theta")

    def phi(self):
        return self.dragon.joint_position("L" + str(self.index) + "_vec_phi")

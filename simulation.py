from Dragon import Dragon
import threading

def sim_loop(dragon):

    while True:
        dragon.set_joint_pos("joint3_pitch", 0)
        dragon.set_joint_pos("joint2_pitch", 0)
        dragon.set_joint_pos("joint1_pitch", 0)
        dragon.set_joint_pos("joint3_yaw", 0)
        dragon.set_joint_pos("joint2_yaw", 0)
        dragon.set_joint_pos("joint1_yaw", 0)
        dragon.step()
        dragon.hover()

def main():
    dragon = Dragon("simple_dragon.urdf")

    threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
    dragon.animate()

if __name__ == "__main__":
    main()

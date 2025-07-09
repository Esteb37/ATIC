from Dragon import Dragon
import threading

def sim_loop(dragon):


    dragon.set_joint_pos("joint3_pitch", 0)
    dragon.set_joint_pos("joint2_pitch", 0)
    dragon.set_joint_pos("joint1_pitch", 0)
    dragon.set_joint_pos("joint3_yaw", 1.5)
    dragon.set_joint_pos("joint2_yaw", 1.5)
    dragon.set_joint_pos("joint1_yaw", 1.5)


    thrust = 9.81 * dragon.total_mass / 8
    while True:
        dragon.step()
        dragon.thrust([thrust] * 8)

def main():
    dragon = Dragon()

    threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
    dragon.animate()

if __name__ == "__main__":
    main()

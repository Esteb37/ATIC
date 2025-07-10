from Dragon import Dragon
import threading

def sim_loop(dragon):

    while True:
        dragon.step()
        input()
        # dragon.hover()

def main():
    dragon = Dragon("dragon.urdf")

    threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
    dragon.animate()

if __name__ == "__main__":
    main()

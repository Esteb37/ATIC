from sim.Dragon import Dragon
import threading

def sim_loop(dragon: Dragon):
    while True:
        dragon.step()
        dragon.hover()

def main():
    dragon = Dragon("dragon.urdf")

    threading.Thread(target=sim_loop, args=(dragon,), daemon=True).start()
    dragon.animate()

if __name__ == "__main__":
    main()

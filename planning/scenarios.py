import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.Dragon import Dragon
import numpy as np

dragon = Dragon()
ell = dragon.MODULE_DISTANCE

USHAPE_5 = {
  "savefile":"ushape_5",

  "obstacles":[{"center": np.array([1, 1, -0.3]), "radius": 0.5}],

  "x_ref":np.array([[3 + ell, 0, 0],
                        [3, 0, 0],
                        [3, ell, 0],
                        [3, 2 * ell, 0],
                        [3 + ell, 2 * ell, 0]
                        ]),



  "x_current":np.array([[0, i * ell, 0] for i in range(5)]),

  "N_drones":5,
  "K_admm":50,
  "T_sim":60,
}

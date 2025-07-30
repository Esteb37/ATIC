import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.Dragon import Dragon
import numpy as np

dragon = Dragon()
ell = dragon.MODULE_DISTANCE

USHAPE_5 = {
  "savefile":"ushape_5",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.5]), "radius": 0.8}],
  "x_ref":np.array([[4 + ell, 0, 0],
                    [4, 0, 0],
                    [4, ell, 0],
                    [4, 2 * ell, 0],
                    [4 + ell, 2 * ell, 0]
                  ]),
  "x_current":np.array([[0, i * ell, 0] for i in range(5)]),
  "N_drones":5,
  "K_admm":50,
  "T_sim":60,
  "u_max": 1.0,
  "gamma": 3,
}

USHAPE_TWO_OBS = {
  "savefile":"ushape_two_obs",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.5]), "radius": 0.8},
                {"center": np.array([3.5, 1.5, -0.5]), "radius": 0.8}
                ],
  "x_ref":np.array([[5 + ell, 0, 0],
                    [5, 0, 0],
                    [5, ell, 0],
                    [5, 2 * ell, 0],
                    [5 + ell, 2 * ell, 0]
                  ]),
  "x_current":np.array([[0, i * ell, 0] for i in range(5)]),
  "N_drones":5,
  "K_admm":50,
  "T_sim":70,
  "u_max": 1.0,
  "gamma": 3,
}

USHAPE_5_SLOW = {
  "savefile":"ushape_5_slow",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.5]), "radius": 0.8}],
  "x_ref":np.array([[4 + ell, 0, 0],
                    [4, 0, 0],
                    [4, ell, 0],
                    [4, 2 * ell, 0],
                    [4 + ell, 2 * ell, 0]
                  ]),
  "x_current":np.array([[0, i * ell, 0] for i in range(5)]),
  "N_drones":5,
  "K_admm":50,
  "T_sim":80,
  "u_max": 0.8,
  "gamma": 3,
}

USHAPE_5_REV = {
  "savefile":"ushape_5_REV",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.5]), "radius": 0.8}],
  "x_ref":np.array([[4, 0, 0],
                    [4 + ell, 0, 0],
                    [4 + ell, ell, 0],
                    [4 + ell, 2 * ell, 0],
                    [4, 2 * ell, 0]
                  ]),
  "x_current":np.array([[0, i * ell, 0] for i in range(5)]),
  "N_drones":5,
  "K_admm":50,
  "T_sim":60,
  "u_max": 1.0,
  "gamma": 3,
}

LINE_9 = {
  "savefile":"line_9",
  "obstacles": [{"center": np.array([2.5, 2.5, -0.5]), "radius": 0.8}],
  "x_ref":np.array([[i * ell, 0.0, 0.0] for i in range(9)]),
  "x_current":np.array([[0, i * ell, 0] for i in range(9)]),
  "N_drones":9,
  "K_admm":100,
  "T_sim":90,
  "u_max": 1.0,
  "gamma": 1.5,
}

USHAPE_9 = {
  "savefile":"ushape_9",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.5]), "radius": 0.7}],
  "x_ref":np.array([[4 + 3 * ell, 0, 0],
                    [4 + 2 * ell, 0, 0],
                    [4 + ell, 0, 0],
                    [4, 0, 0],
                    [4, ell, 0],
                    [4, 2 * ell, 0],
                    [4 + ell, 2 * ell, 0],
                    [4 + 2 * ell, 2 * ell, 0],
                    [4 + 3 * ell, 2 * ell, 0],
                  ]),

  "x_current":np.array([[0, i * ell, 0] for i in range(9)]),
  "N_drones":9,
  "K_admm":50,
  "T_sim":80,
  "u_max": 1.0,
  "gamma": 1,
}

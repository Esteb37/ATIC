import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sim.Dragon import Dragon
import numpy as np

dragon = Dragon()
ell = dragon.MODULE_DISTANCE
dist = ell * np.sqrt(2) / 2

USHAPE_5 = {
  "savefile":"ushape_5",
  "obstacles": [],
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
  "rho": 15,
}

SNAKE_5 = {
  "savefile":"snake_5",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.5]), "radius": 0.8},
                {"center": np.array([3.5, 1.5, -0.5]), "radius": 0.8}
                ],
  "x_ref":np.array([[4, 0, 0],
                    [4 + dist, dist, 0],
                    [4, 2 * dist, 0],
                    [4 + dist, 3 * dist, 0],
                    [4, 4 * dist, 0]
                  ]),
  "x_current":np.array([[0, i * ell, 0] for i in range(5)]),
  "N_drones":5,
  "K_admm":50,
  "T_sim":60,
  "u_max": 1.0,
  "gamma": 3,
  "rho": 15,
}

SNAKE_5_LONG = {
  "savefile":"snake_5_long",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.3]), "radius": 0.8},
                {"center": np.array([3.5, 1.5, -0.3]), "radius": 0.8},
                {"center": np.array([5.5, 0.5, -0.3]), "radius": 0.8},
                {"center": np.array([7.5, 1.5, 0.3]), "radius": 0.8},
                {"center": np.array([9.5, 2.5, 0.3]), "radius": 0.8},
                ],

  "x_ref":np.array([[11, 0, 0],
                    [11 + dist, dist, 0],
                    [11, 2 * dist, 0],
                    [11 + dist, 3 * dist, 0],
                    [11, 4 * dist, 0]
                  ]),

  "x_current":np.array([[0, i * ell, 0] for i in range(5)]),
  "N_drones":5,
  "K_admm":50,
  "T_sim":120,
  "u_max": 1.0,
  "gamma": 3,
  "rho": 15,
}

LINE_5_LONG = {
  "savefile":"line_5_long",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.3]), "radius": 0.8},
                {"center": np.array([3.5, 1.5, -0.3]), "radius": 0.8},
                {"center": np.array([5.5, 0.5, -0.3]), "radius": 0.8},
                {"center": np.array([7.5, 1.5, 0.3]), "radius": 0.8},
                {"center": np.array([9.5, 2.5, 0.3]), "radius": 0.8},
                ],

  "x_ref":np.array([[0, 11 + i * ell, 0] for i in range(5)]),
  "x_current":np.array([[0, i * ell, 0] for i in range(5)]),
  "N_drones":5,
  "K_admm":50,
  "T_sim":120,
  "u_max": 1.0,
  "gamma": 3,
  "rho": 15,
}

USHAPE_TWO_OBS = {
  "savefile":"ushape_two_obs",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.5]), "radius": 0.8},
                {"center": np.array([3.5, 1.5, -0.5]), "radius": 0.8}
                ],
  "x_ref":np.array([[6 + ell, 0, 0],
                    [6, 0, 0],
                    [6, ell, 0],
                    [6, 2 * ell, 0],
                    [6 + ell, 2 * ell, 0]
                  ]),
  "x_current":np.array([[0, i * ell, 0] for i in range(5)]),
  "N_drones":5,
  "K_admm":50,
  "T_sim":80,
  "u_max": 1.0,
  "gamma": 3,
  "rho": 15.0,
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
  "rho": 15,
}

USHAPE_5_REV = {
  "savefile":"ushape_5_REV",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.5]), "radius": 0.8}],
  "x_current":np.array([[5 + ell, 0, 0],
                    [5, 0, 0],
                    [5, ell, 0],
                    [5, 2 * ell, 0],
                    [5 + ell, 2 * ell, 0]
                  ]),
  "x_ref":np.array([[0, i * ell, 0] for i in range(5)]),
  "N_drones":5,
  "K_admm":50,
  "T_sim":60,
  "u_max": 1.0,
  "gamma": 3,
  "rho": 15,
}

LINE_9 = {
  "savefile":"line_9",
  "obstacles": [],
  "x_ref":np.array([[i * ell, 0.0, 0.0] for i in range(9)]),
  "x_current":np.array([[0, i * ell, 0] for i in range(9)]),
  "N_drones":9,
  "K_admm":50,
  "T_sim":90,
  "u_max": 1.0,
  "gamma": 2,
  "rho": 15,
}

LINE_9_NO = {
  "savefile":"line_9_no",
  "obstacles": [],
  "x_ref":np.array([[i * ell, 0.0, 0.0] for i in range(9)]),
  "x_current":np.array([[0, i * ell, 0] for i in range(9)]),
  "N_drones":9,
  "K_admm":50,
  "T_sim":80,
  "u_max": 1.0,
  "gamma": 1.5,
  "rho": 15,
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
  "gamma": 1.5,
  "rho": 15,
}

USHAPE_9_NO = {
  "savefile":"ushape_9_no",
  "obstacles": [],
  "x_ref":np.array([[4 + 2 * ell, 0, 0],
                    [4 + ell, 0, 0],
                    [4, 0, 0],
                    [4, ell, 0],
                    [4, 2 * ell, 0],
                    [4, 3 * ell, 0],
                    [4 + ell, 3 * ell, 0],
                    [4 + 2 * ell, 3 * ell, 0],
                    [4 +  3 * ell, 3 * ell, 0],
                  ]),

  "x_current":np.array([[0, i * ell, 0] for i in range(9)]),
  "N_drones":9,
  "K_admm":50,
  "T_sim":90,
  "u_max": 1.0,
  "gamma": 2,
  "rho": 15,
}

NONAGON = {
  "savefile":"nonagon",
  "obstacles": [{"center": np.array([1.5, 2.5, -0.5]), "radius": 0.7}],
  "x_ref":np.array([[4.0, 1.0, 0.0],
                    [4.766044443118978, 0.35721239031346075, 0.0],
                    [5.766044443118978, 0.35721239031346075, 0.0],
                    [6.532088886237956, 0.9999999999999999, 0.0],
                    [6.705737063904886, 1.9848077530122077, 0.0],
                    [6.205737063904886, 2.850833156796646, 0.0],
                    [5.266044443118978, 3.192853300122315, 0.0],
                    [4.32635182233307, 2.8508331567966465, 0.0],
                    [3.82635182233307, 1.9848077530122086, 0.0]]),
  "x_current":np.array([[0, i * ell, 0] for i in range(9)]),
  "N_drones":9,
  "K_admm":50,
  "T_sim":80,
  "u_max": 1.0,
  "gamma": 1,
  "rho": 15,
}

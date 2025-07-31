# ADMM-based Distributed MPC for Modular Articulated Aerial Robots

State-of-the-art articulated modular aerial robots provide versatility in wrench execution and aerial manipulation through customizable shape formations and vectorizable thrust control. The DRAGON robot is one of these systems, comprised of four modules interlinked with two-DoF joints for custom shape selection and a vectoring apparatus on each module for precise thrust control. Nevertheless, the high number of degrees of freedom makes centralized model-based planning and control mechanisms computationally heavy and fundamentally unscalable. In this paper, we propose a decentralized, distributed, scalable solution for path planning that spreads the computational load across all modules by treating them as independent, rigidly linked drones and solving a distributed, constrained MPC problem with ADMM.

![Gif](saves/ushape_two_obs_simulation.gif)
![Gif](saves/snake_5_long_simulation.gif)
![Gif](saves/line_9_simulation.gif)
![Gif](saves/ushape_9_no_simulation.gif)
![Gif](saves/ushape_5_REV_simulation.gif)
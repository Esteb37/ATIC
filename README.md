# ADMM-based Distributed MPC for Modular Articulated Aerial Robots

State-of-the-art articulated modular aerial robots provide versatility in wrench execution and aerial manipulation through customizable shape formations and vectorizable thrust control. The DRAGON robot is one of these systems, comprised of four modules interlinked with two-DoF joints for custom shape selection and a vectoring apparatus on each module for precise thrust control. Nevertheless, the high number of degrees of freedom makes centralized model-based planning and control mechanisms computationally heavy and fundamentally unscalable. In this paper, we propose a decentralized, distributed, scalable solution for path planning that spreads the computational load across all modules by treating them as independent, rigidly linked drones and solving a distributed, constrained MPC problem with ADMM.

<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="saves/snake_5_long_simulation.gif" width="100%"/>
  <img src="saves/ushape_two_obs_simulation.gif" width="49%"/>
  <img src="saves/line_9_simulation.gif" width="49%"/>
  <img src="saves/ushape_9_no_simulation.gif" width="49%"/>
  <img src="saves/ushape_5_REV_simulation.gif" width="49%"/>
</div>

# Distributed Thrust Allocation for Wrench Control


<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <div style="width: 49%; text-align:center">
  <h2>Open Loop Thrust</h2>
  </div>
  <div style="width: 49%; text-align:center">
  <h2>Distributed Thrust Allocation</h2>
  </div>
  <img src="saves/no_control_wrench.gif" width="49%"/>
  <img src="saves/forward_snake_wrench.gif" width="49%"/>
</div>

## More examples
<div style="display: flex; flex-wrap: wrap; gap: 10px;">
  <img src="saves/long_snake_wrench.gif" width="100%"/>
  <img src="saves/hover_ushape_wrench.gif" width="49%"/>
  <img src="saves/turn_line_wrench.gif" width="49%"/>
</div>

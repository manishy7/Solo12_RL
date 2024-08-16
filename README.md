# Solo12_RL
The Solo12 project focuses on quadrupedal balancing and robot locomotion using reinforcement learning, ZMP control, and PD controllers.
Running the Main Simulation
### PyBullet Installation:
- PyBullet is the physics engine used for simulation. It can be installed via pip:

```bash
pip install pybullet
```
### Gymnasium Installation:
- The Gymnasium library is used for creating and managing the RL environment. It can be installed using:
```bash
pip install gymnasium==0.26.0
```
### Stable Baselines3:
- The Stable Baselines3 library is used for the implementation of the PPO algorithm:
```bash
pip install stable-baselines3[extra]
```

### Running the Main Simulation
1. To start the reinforcement learning simulation with ZMP and PD control, run:
```bash
python3 solo12_main_rl_zmp.py
```
2. Simuation and Testing
```bash
cd graphsview
python solo12_test1.py   # For basic terrain
python solo12_terrain_view.py  # For different terrains

```

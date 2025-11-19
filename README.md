# Swarm Rovers Mapping with GAN (ROS 2 Humble)

This project implements a multi-robot autonomous mapping system using ROS 2 Humble and Gazebo 11 with swarm rovers equipped with LiDAR sensors. It uses SLAM Toolbox for individual robot mapping, merges the maps, and integrates a Generative Adversarial Network (GAN) to clean and refine the merged occupancy grid map in real-time.

---

## Features

- **Multi-robot SLAM:** Three independent rovers explore and map the environment simultaneously.
- **Map merger:** Combines individual rover maps into a single unified map.
- **Dataset generator:** Automatically creates training pairs (noisy and clean maps) during exploration.
- **Deep learning integration:** A GAN is trained online to denoise and complete maps.
- **Full ROS 2 ecosystem:** All nodes, topics, and launches use ROS 2 standards.
- **Gazebo simulation:** Real-time 3D robot simulation environment.

---

## Project Structure
```
swarm_gan_slam_ws/
├── src/
│   └── swarm_gan_slam/
│       ├── launch/
│       │   └── swarm_launch.py          # Main launch file to start Gazebo, rovers, SLAM, GAN, etc.
│       ├── swarm_gan_slam/
│       │   ├── rover_controller.py      # Rover autonomous navigation & obstacle avoidance
│       │   ├── map_merger.py            # Merges rover's maps into one
│       │   ├── dataset_generator.py     # Generates training dataset automatically
│       │   ├── gan_model.py             # GAN architecture and training functions
│       │   └── gan_processor.py         # Runs GAN training and cleans maps online
│       ├── models/
│       │   └── rover.sdf                # Robot model description with sensors and plugins
│       ├── worlds/
│       │   └── swarm_world.world        # Gazebo simulation environment
│       ├── urdf/
│       │   └── rover.urdf               # Robot description file
│       ├── package.xml                  # ROS 2 package info and dependencies
│       └── setup.py                     # Python package setup
└── install/                             # Installed/compiled workspace files
```

---

## Usage

1. **Build and source the workspace:**
```bash
   cd ~/swarm_gan_slam_ws
   colcon build --symlink-install
   source install/setup.bash
```

2. **Launch the entire system** (spawns Gazebo, rovers, SLAM, GAN, etc.):
```bash
   ros2 launch swarm_gan_slam swarm_launch.py
```

3. **In a new terminal, start RViz2 for visualization:**
```bash
   rviz2
```

4. **In RViz2:**
   - Set Fixed Frame to `map`
   - Add Map displays for `/rover1/map`, `/rover2/map`, `/rover3/map`, `/merged_map`, and `/cleaned_map`
   - Add LaserScan displays for `/rover1/scan`, `/rover2/scan`, `/rover3/scan`
   - Optionally add Grid and TF visualization for better context

5. Watch the robots autonomously explore and ensure maps build up progressively.

---

## Current Limitations

- The GAN integration is **added but not yet fully effective**. The training continues but the cleaned maps might not significantly improve over merged maps yet. The GAN training is experimental and may require further tuning to stabilize and produce better results.
- You may notice the cleaned map sometimes looks similar to the merged map or with minor improvements.
- The project is designed as a proof-of-concept for integrating deep learning with multi-robot SLAM.

---

## Notes on Implementation

- **Swarm Approach:** Using three rovers speeds up exploration and creates richer datasets.
- **Map Merger:** Simple averaging of available occupancy grid data from individual maps.
- **Dataset generator:** Creates noisy/clean pairs using self-supervised methods without any external data.
- **GAN Model:** Encoder-decoder architecture trained online with L1 and adversarial losses.
- **Simulation:** Gazebo world can be replaced with different environments, including mazes.

---

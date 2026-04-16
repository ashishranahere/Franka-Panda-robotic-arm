Hybrid APF-Guided RRT Motion Planning and Autonomous Pick-and-Place

1. Overview
This project implements an integrated robotic system using the Franka Panda manipulator in the PyBullet simulation environment. The work addresses two key objectives:

1. Development of a vision-based autonomous pick-and-place pipeline.
2. Design and evaluation of a hybrid motion planner combining Artificial Potential Fields (APF) with Rapidly-exploring Random Trees (RRT), including an optimization-based enhancement.

The system demonstrates perception, motion planning, and control in a cluttered three-dimensional workspace.

2. System Components

2.1 Perception and Grasping
- Overhead and wrist-mounted camera integration
- Pixel-to-world coordinate transformation
- Object detection via segmentation and contour-based heuristics
- Automated grasp pipeline: approach, descend, grasp, lift

2.2 Motion Planning

Baseline (APF-Guided RRT)
- Goal-biased sampling
- Attractive and repulsive potential fields
- Collision-aware expansion

Enhancement (Optimization-Based Smoothing)
- Shortcut-based path smoothing
- Reduced path length and node count

Local Minima Handling
- Virtual Obstacle Method (VOM)

3. Experimental Evaluation

Run 1:
Success Rate: 85.0%
Avg Time: 0.457 s
Path Length: 7.94 → 5.04
Nodes: 182 → 2

Run 2:
Success Rate: 85.0%
Avg Time: 0.502 s
Path Length: 8.08 → 5.35
Nodes: 192 → 2

4. Analysis
The enhanced planner reduces path length and node count significantly, with a slight increase in computation time. The hybrid approach ensures robustness in cluttered environments.

5. Outputs
- benchmark_results.csv
- metrics_comparison.png
- rrt_path_visualization.png

6. Execution
pip install pybullet numpy opencv-python matplotlib

python3 7.1APF.py
python3 8APF.py

7. Conclusion
The system successfully integrates perception and planning, producing efficient and robust motion in complex environments.

Reference work-- 

D. Prajapati, A. Mehra, P. Kumar, A. Rana, S. K. Surya Prakash and A. Shukla, "MPPF: Mobile Probe Potential Field for Real-Time Multi-Agent Navigation in Industrial Environments," in IEEE Access, vol. 14, pp. 11084-11095, 2026, doi: 10.1109/ACCESS.2026.3655886.

Author: Ashish

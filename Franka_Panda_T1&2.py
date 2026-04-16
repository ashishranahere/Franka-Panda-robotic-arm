import pybullet as p
import pybullet_data
import time
import numpy as np
import random
import cv2
import math

# --- APF TUNING CONSTANTS ---
ATTRACTION_GAIN = 1.5
REPULSION_GAIN = 0.05
REPULSION_THRESHOLD = 0.20

# --- FRANKA PANDA KINEMATIC LIMITS ---
LOWER_LIMITS = [-2.89, -1.76, -2.89, -3.07, -2.89, -0.01, -2.89]
UPPER_LIMITS = [2.89, 1.76, 2.89, -0.06, 2.89, 3.75, 2.89]
JOINT_RANGES = [5.78, 3.52, 5.78, 3.01, 5.78, 3.76, 5.78]
REST_POSES = [0, -0.4, 0, -2.4, 0, 2.0, 0.8]

def start_simulation(gui=True):
    p.connect(p.GUI if gui else p.DIRECT)
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0) 
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)

    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=1.0) 

    robot_id = p.loadURDF("franka_panda/panda.urdf", basePosition=[0, 0, 0], useFixedBase=True)
    p.changeDynamics(robot_id, 9, lateralFriction=2.0, spinningFriction=1.0)
    p.changeDynamics(robot_id, 10, lateralFriction=2.0, spinningFriction=1.0)

    return plane_id, robot_id

def spawn_dynamic_obstacles():
    obstacles = []
    
    # Distributed across all 4 quadrants to create a 360-degree cluttered canopy.
    obstacle_configs = [
        (p.GEOM_CYLINDER, 0.03, 0.30, [0.35, 0.3, 0.15]),
        (p.GEOM_SPHERE, 0.04, None, [0.5, 0.4, 0.1]),
        (p.GEOM_CYLINDER, 0.02, 0.40, [0.4, -0.3, 0.20]),
        (p.GEOM_SPHERE, 0.035, None, [0.6, -0.2, 0.25]),
        (p.GEOM_CYLINDER, 0.04, 0.25, [-0.3, 0.4, 0.125]),
        (p.GEOM_SPHERE, 0.05, None, [-0.5, 0.2, 0.05]),
        (p.GEOM_CYLINDER, 0.03, 0.35, [-0.4, -0.4, 0.175])
    ]

    for config in obstacle_configs:
        geom_type, radius, length, pos = config
        color = [random.uniform(0.6, 1.0), random.uniform(0.0, 0.2), random.uniform(0.0, 0.2), 1]
        
        if geom_type == p.GEOM_CYLINDER:
            visual_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=radius, length=length, rgbaColor=color)
            collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=radius, height=length)
        else:
            visual_shape = p.createVisualShape(p.GEOM_SPHERE, radius=radius, rgbaColor=color)
            collision_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=radius)

        obs_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=visual_shape,
            baseCollisionShapeIndex=collision_shape,
            basePosition=pos)
        obstacles.append(obs_id)

    return obstacles

# --- TRUE RRT TREE BUILDER ---
class RRTNode:
    def __init__(self, q):
        self.q = np.array(q)
        self.parent = None

class RRT:
    def __init__(self, robot_id, obstacles, start_q, goal_q, max_iter=500, step_size=0.2, goal_sample_rate=0.1):
        self.robot_id = robot_id
        self.obstacles = obstacles
        self.start_node = RRTNode(start_q)
        self.goal_node = RRTNode(goal_q)
        self.node_list = [self.start_node]
        self.max_iter = max_iter
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate

    def check_collision(self, q):
        state_id = p.saveState()
        for i in range(7): p.resetJointState(self.robot_id, i, q[i])
        p.stepSimulation()
        
        collision = False
        for obs in self.obstacles:
            contacts = p.getClosestPoints(self.robot_id, obs, distance=0.0)
            if len(contacts) > 0:
                collision = True
                break
                
        p.restoreState(state_id)
        p.removeState(state_id)
        return collision

    def plan(self):
        for i in range(self.max_iter):
            if random.random() < self.goal_sample_rate:
                rnd_q = self.goal_node.q
            else:
                rnd_q = np.array([random.uniform(LOWER_LIMITS[j], UPPER_LIMITS[j]) for j in range(7)])

            nearest_ind = np.argmin([np.linalg.norm(node.q - rnd_q) for node in self.node_list])
            nearest_node = self.node_list[nearest_ind]

            direction = rnd_q - nearest_node.q
            dist = np.linalg.norm(direction)
            if dist == 0: continue
            direction = direction / dist
            new_q = nearest_node.q + direction * min(self.step_size, dist)

            if not self.check_collision(new_q):
                new_node = RRTNode(new_q)
                new_node.parent = nearest_node
                self.node_list.append(new_node)

                if np.linalg.norm(new_node.q - self.goal_node.q) <= self.step_size:
                    final_node = RRTNode(self.goal_node.q)
                    final_node.parent = new_node
                    self.node_list.append(final_node)
                    return self.extract_path(final_node)
        return None

    def extract_path(self, node):
        path = []
        while node is not None:
            path.append(node.q)
            node = node.parent
        return path[::-1] 

# --- 3D CARTESIAN APF LOGIC ---
def calculate_total_force_3d(robot_id, ee_link_index, current_ee_pos, goal_pos, obstacle_ids):
    disp = np.array(goal_pos) - np.array(current_ee_pos)
    dist_goal = np.linalg.norm(disp)
    dir_att = disp / dist_goal if dist_goal > 0 else np.zeros(3)
    f_att = ATTRACTION_GAIN * disp

    f_rep_total = np.zeros(3)
    for obs in obstacle_ids:
        contacts = p.getClosestPoints(bodyA=robot_id, bodyB=obs, distance=REPULSION_THRESHOLD, linkIndexA=ee_link_index)
        for pt in contacts:
            pos_robot = np.array(pt[5])
            pos_obs = np.array(pt[6])
            dist = pt[8]
            if 0 < dist < REPULSION_THRESHOLD:
                dir_rep = (pos_robot - pos_obs) / dist
                mag = REPULSION_GAIN * ((1.0/dist) - (1.0/REPULSION_THRESHOLD)) * (1.0/(dist**2))
                f_rep_total += mag * dir_rep

    f_total = f_att + f_rep_total
    
    mag_att = np.linalg.norm(f_att)
    mag_rep = np.linalg.norm(f_rep_total)
    dir_att_n = f_att / mag_att if mag_att > 0 else np.zeros(3)
    dir_rep_n = f_rep_total / mag_rep if mag_rep > 0 else np.zeros(3)
    alignment = np.dot(dir_att_n, dir_rep_n)

    in_local_minima_confirmed = False
    vom_kick_applied = False
    
    if alignment < -0.95 and mag_rep > (mag_att * 0.5):
        vom_kick_applied = True
        v_perp = np.cross(f_att, f_rep_total)
        if np.linalg.norm(v_perp) < 1e-5:
            v_dummy = np.array([0, 0, 1]) 
            if abs(np.dot(dir_rep_n, v_dummy)) > 0.9:
                v_dummy = np.array([0, 1, 0])
            v_perp = np.cross(f_rep_total, v_dummy)
        
        v_perp_n = v_perp / np.linalg.norm(v_perp)
        f_vom = (ATTRACTION_GAIN * 1.5) * v_perp_n
        f_total += f_vom
    
    f_total_mag = np.linalg.norm(f_total)
    if f_total_mag < 0.1:
        in_local_minima_confirmed = True

    metrics = {'mag_att': mag_att, 'mag_rep': mag_rep, 'alignment': alignment, 'vom_kick_applied': vom_kick_applied}
    return f_total, metrics, in_local_minima_confirmed

# --- HYBRID MOVEMENT CONTROLLER ---
def move_to_position(robot_id, target_pos, obstacles=[], use_apf=False, strict_orientation=True):
    end_effector_index = 11

    if strict_orientation:
        orientation = p.getQuaternionFromEuler([3.1415, 0, 0])
        goal_q = p.calculateInverseKinematics(
            robot_id, end_effector_index, target_pos, targetOrientation=orientation,
            lowerLimits=LOWER_LIMITS, upperLimits=UPPER_LIMITS, jointRanges=JOINT_RANGES, restPoses=REST_POSES,
            maxNumIterations=200, residualThreshold=1e-4)
    else:
        goal_q = p.calculateInverseKinematics(
            robot_id, end_effector_index, target_pos,
            lowerLimits=LOWER_LIMITS, upperLimits=UPPER_LIMITS, jointRanges=JOINT_RANGES, restPoses=REST_POSES,
            maxNumIterations=200, residualThreshold=1e-4)

    if use_apf and len(obstacles) > 0:
        ee_state = p.getLinkState(robot_id, end_effector_index)
        current_ee_pos = ee_state[0]

        f_total, metrics, stuck = calculate_total_force_3d(robot_id, end_effector_index, current_ee_pos, target_pos, obstacles)
        step_size = 0.05
        next_target_pos = np.array(current_ee_pos) + (f_total * step_size)

        if strict_orientation:
            next_q = p.calculateInverseKinematics(
                robot_id, end_effector_index, next_target_pos.tolist(), targetOrientation=orientation,
                lowerLimits=LOWER_LIMITS, upperLimits=UPPER_LIMITS, jointRanges=JOINT_RANGES, restPoses=REST_POSES,
                maxNumIterations=200)
        else:
            next_q = p.calculateInverseKinematics(
                robot_id, end_effector_index, next_target_pos.tolist(),
                lowerLimits=LOWER_LIMITS, upperLimits=UPPER_LIMITS, jointRanges=JOINT_RANGES, restPoses=REST_POSES,
                maxNumIterations=200)

        for i in range(7):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=next_q[i], force=150, maxVelocity=1.5)
        return metrics, stuck
    else:
        for i in range(7):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=goal_q[i], force=200, maxVelocity=1.0)
        return None, False

def reset_to_home_pose(robot_id):
    for i in range(7):
        p.resetJointState(robot_id, i, REST_POSES[i])
        p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, targetPosition=REST_POSES[i], force=200)
    for joint in [9, 10]:
        p.resetJointState(robot_id, joint, 0.04)
        p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, targetPosition=0.04, force=50)

def control_gripper(robot_id, open_gripper=True):
    target = 0.04 if open_gripper else 0.00 
    force = 100 if not open_gripper else 50 
    for joint in [9, 10]:
        p.setJointMotorControl2(robot_id, joint, p.POSITION_CONTROL, targetPosition=target, force=force)

def get_ee_distance_to_target(robot_id, target_pos):
    ee_state = p.getLinkState(robot_id, 11)
    return np.linalg.norm(np.array(ee_state[0]) - np.array(target_pos))

# --- CAMERA / VISION PIPELINE ---
def display_camera_feed(window_name, rgb_data, width, height):
    img_rgba = np.array(rgb_data).reshape((height, width, 4)).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2BGR)
    cv2.imshow(window_name, img_bgr)
    cv2.waitKey(1)

def spawn_cubes(num_cubes=3):
    cube_ids = []
    for i in range(num_cubes):
        x = random.uniform(0.5, 0.7)
        y = random.uniform(-0.2, 0.2)
        z = 0.05
        cube_id = p.loadURDF("cube_small.urdf", basePosition=[x, y, z])
        color = [random.random(), random.random(), random.random(), 1]
        p.changeVisualShape(cube_id, -1, rgbaColor=color)
        p.changeDynamics(cube_id, -1, mass=0.1, lateralFriction=2.0, rollingFriction=0.01)
        cube_ids.append(cube_id)
    return cube_ids

def get_overhead_camera_data():
    width, height = 640, 480
    view_matrix = p.computeViewMatrix(cameraEyePosition=[0.6, 0, 1.0], cameraTargetPosition=[0.6, 0, 0], cameraUpVector=[0, 1, 0])
    projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=width/height, nearVal=0.1, farVal=2.0)
    width, height, rgb, depth_buffer, seg = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return width, height, view_matrix, projection_matrix, depth_buffer, seg, rgb

def pixel_to_world(u, v, depth_buffer, view_matrix, width, height):
    z_b = depth_buffer[v][u]
    near, far = 0.1, 2.0
    z_ndc = 2.0 * z_b - 1.0
    d = (2.0 * near * far) / (far + near - z_ndc * (far - near))
    fov_rad = np.radians(60) 
    f = (height / 2.0) / np.tan(fov_rad / 2.0)
    cx, cy = width / 2.0, height / 2.0
    K_inv = np.linalg.inv(np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]]))
    P_pixel = np.array([u, v, 1.0])
    P_cam = K_inv @ P_pixel * d
    P_cam_gl = np.array([P_cam[0], -P_cam[1], -P_cam[2], 1.0])
    view_mat = np.array(view_matrix).reshape(4, 4).T
    T_camera_world = np.linalg.inv(view_mat)
    P_world = T_camera_world @ P_cam_gl
    return P_world[:3]

def get_cube_pose_from_vision(target_cube_id):
    """OpenCV Heuristic: Extracts centroid and physical orientation (yaw)."""
    w, h, vm, pm, depth, seg, rgb = get_overhead_camera_data()
    mask = np.uint8(seg == target_cube_id) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None
        
    largest_contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest_contour)
    (u, v), (width, height), angle_deg = rect
    
    target_yaw = math.radians(angle_deg)
    pos_3d = pixel_to_world(int(u), int(v), depth, vm, w, h)
    
    img_bgr = cv2.cvtColor(np.array(rgb).reshape((h, w, 4)).astype(np.uint8), cv2.COLOR_RGBA2BGR)
    box = cv2.boxPoints(rect)
    box = np.int32(box)
    cv2.drawContours(img_bgr, [box], 0, (0, 255, 0), 2)
    cv2.circle(img_bgr, (int(u), int(v)), 4, (0, 0, 255), -1)
    cv2.putText(img_bgr, f"Yaw: {angle_deg:.1f} deg", (int(u)-40, int(v)-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.imshow("Heuristic Object Classification", img_bgr)
    cv2.waitKey(1)
    
    return pos_3d, target_yaw

def get_wrist_camera_data(robot_id):
    width, height = 320, 240 
    link_state = p.getLinkState(robot_id, 11)
    ee_pos = link_state[0]
    rot_matrix = np.array(p.getMatrixFromQuaternion(link_state[1])).reshape(3, 3)
    camera_vector = rot_matrix[:, 2] 
    up_vector = rot_matrix[:, 1]
    target_pos = ee_pos + (camera_vector * 0.1)
    view_matrix = p.computeViewMatrix(cameraEyePosition=ee_pos, cameraTargetPosition=target_pos, cameraUpVector=up_vector)
    
    # FIX: Changed farVal to 2.0 so the floor actually renders when the arm is high up
    projection_matrix = p.computeProjectionMatrixFOV(fov=60, aspect=width/height, nearVal=0.01, farVal=2.0)
    
    width, height, rgb, depth_buffer, seg = p.getCameraImage(width, height, view_matrix, projection_matrix, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    return width, height, rgb, seg

def _move_until_reached(robot_id, target_pos, target_orientation):
    ee_index = 11
    max_steps = 400 
    for _ in range(max_steps):
        target_q = p.calculateInverseKinematics(
            robot_id, ee_index, target_pos, targetOrientation=target_orientation,
            lowerLimits=LOWER_LIMITS, upperLimits=UPPER_LIMITS, 
            jointRanges=JOINT_RANGES, restPoses=REST_POSES,
            maxNumIterations=100, residualThreshold=1e-4)
        for i in range(7):
            p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 
                                    targetPosition=target_q[i], force=150, maxVelocity=1.0)
        p.stepSimulation()
        time.sleep(1/240.0)
        current_pos = p.getLinkState(robot_id, ee_index)[0]
        if np.linalg.norm(np.array(current_pos) - np.array(target_pos)) < 0.02:
            break

def grasp_point_world(robot_id, cube_id, target_pos, target_yaw, obstacles):
    print(f"Initiating automated grasp pipeline (Aligned at {math.degrees(target_yaw):.1f} deg)...")
    
    closest_obs_dist = 1.0
    escape_vector = np.array([0, 0, 1]) 
    
    for obs in obstacles:
        pts = p.getClosestPoints(bodyA=cube_id, bodyB=obs, distance=0.15)
        if pts and pts[0][8] < closest_obs_dist:
            closest_obs_dist = pts[0][8]
            obs_pos = np.array(pts[0][6])
            escape_vector = np.array(target_pos) - obs_pos
            escape_vector[2] = 0.2 
            escape_vector /= np.linalg.norm(escape_vector)

    if closest_obs_dist < 0.12:
        print(f"⚠️ Near-Obstacle Detect! Using {closest_obs_dist:.2f}m side-approach.")
        yaw = np.arctan2(escape_vector[1], escape_vector[0])
        target_orientation = p.getQuaternionFromEuler([3.14, 0.5, yaw])
    else:
        target_orientation = p.getQuaternionFromEuler([3.14, 0, target_yaw])

    _move_until_reached(robot_id, target_pos + escape_vector * 0.15, target_orientation)
    descend_pos = [target_pos[0], target_pos[1], target_pos[2] - 0.01]
    _move_until_reached(robot_id, descend_pos, target_orientation)
    
    control_gripper(robot_id, open_gripper=False)
    for _ in range(180):
        p.stepSimulation()
        time.sleep(1/240.0)

    contacts = p.getContactPoints(robot_id, cube_id)
    if len(contacts) == 0:
        print("❌ GRASP FAILED: No contact with object. Retrying...")
        control_gripper(robot_id, open_gripper=True)
        return False 

    lift_orientation = p.getQuaternionFromEuler([3.14, 0, 0])
    _move_until_reached(robot_id, [target_pos[0], target_pos[1], 0.5], lift_orientation)
    
    print("✅ Grasp sequence completed successfully.")
    return True

def get_closest_obstacle_distance(robot_id, ee_link_index, obstacle_ids):
    min_dist = 10.0  
    for obs_id in obstacle_ids:
        closest_points = p.getClosestPoints(bodyA=robot_id, bodyB=obs_id, distance=0.2, linkIndexA=ee_link_index)
        if closest_points:
            dist = closest_points[0][8]
            if dist < min_dist:
                min_dist = dist
    return min_dist

def shortcut_path(robot_id, path, obstacles, iterations=50):
    if path is None or len(path) < 3: return path
    smoothed_path = list(path)
    for _ in range(iterations):
        if len(smoothed_path) < 3: break
        idx1 = random.randint(0, len(smoothed_path) - 3)
        idx2 = random.randint(idx1 + 2, len(smoothed_path) - 1)
        q1, q2 = smoothed_path[idx1], smoothed_path[idx2]
        
        is_safe = True
        steps = 10 
        for s in range(1, steps):
            q_interp = q1 + (q2 - q1) * (s / steps)
            state_id = p.saveState()
            for i in range(7): p.resetJointState(robot_id, i, q_interp[i])
            
            collision = False
            for obs in obstacles:
                if len(p.getClosestPoints(robot_id, obs, distance=0.01)) > 0:
                    collision = True
                    break
            
            p.restoreState(state_id)
            p.removeState(state_id)
            if collision:
                is_safe = False
                break
        
        if is_safe:
            smoothed_path = smoothed_path[:idx1 + 1] + smoothed_path[idx2:]
    return smoothed_path

if __name__ == "__main__":
    plane_id, robot_id = start_simulation()
    reset_to_home_pose(robot_id)
    
    cube_ids = spawn_cubes(3)
    obstacles = spawn_dynamic_obstacles()
    cubes_placed = 0 
    
    print("Letting physics settle...")
    for _ in range(240): 
        p.stepSimulation()
        time.sleep(1/240.0)
    
    target_cube = cube_ids[0]
    state = "VISION_DETECT"
    target_3d_pos = None
    sim_step_counter = 0 
    
    stuck_counter = 0
    rrt_path = None
    rrt_path_index = 0
    active_drop_target = None 
    release_timer = 0

    while True:
        if state == "VISION_DETECT":
            target_3d_pos, target_yaw = get_cube_pose_from_vision(target_cube)
            if target_3d_pos is not None:
                grasp_success = grasp_point_world(robot_id, target_cube, target_3d_pos, target_yaw, obstacles)
                if grasp_success:
                    state = "GET_USER_INPUT"
            else:
                time.sleep(0.5)

        elif state == "GET_USER_INPUT":
            print(f"\n📦 Cube {cubes_placed + 1} secured!")
            user_input = input("Enter drop coordinates X and Y separated by a space (e.g., 0.5 -0.3): ")
            try:
                coords = user_input.strip().split()
                if len(coords) == 2:
                    drop_x, drop_y = float(coords[0]), float(coords[1])
                    distance_from_base = (drop_x**2 + drop_y**2) ** 0.5
                    
                    if 0.2 <= distance_from_base <= 0.85:
                        print(f"✅ Target accepted. Moving to [{drop_x}, {drop_y}].")
                        active_drop_target = [drop_x, drop_y, 0.05] 
                        state = "MOVE_TO_APEX" 
                    else:
                        print("❌ LOCATION OUTSIDE WORKSPACE!")
                else:
                    print("⚠️ Invalid format.")
            except ValueError:
                print("⚠️ Invalid input.")

        elif state == "MOVE_TO_APEX":
            current_ee_pos = p.getLinkState(robot_id, 11)[0]
            apex_target = [current_ee_pos[0], current_ee_pos[1], 0.6] 
            print("🚀 Lifting to apex...")
            _move_until_reached(robot_id, apex_target, p.getQuaternionFromEuler([3.1415, 0, 0]))
            if get_ee_distance_to_target(robot_id, apex_target) < 0.05:
                state = "MOVE_ACROSS"

        elif state == "MOVE_ACROSS":
            if rrt_path is not None:
                target_q = rrt_path[rrt_path_index]
                for i in range(7):
                    p.setJointMotorControl2(robot_id, i, p.POSITION_CONTROL, 
                                            targetPosition=target_q[i], force=200, maxVelocity=1.0)
                
                current_q = [p.getJointState(robot_id, i)[0] for i in range(7)]
                if np.linalg.norm(np.array(current_q) - np.array(target_q)) < 0.1:
                    rrt_path_index += 1
                    if rrt_path_index >= len(rrt_path):
                        print("✅ RRT detour complete. Resuming APF.")
                        rrt_path = None
                        stuck_counter = 0
            else:
                target = [active_drop_target[0], active_drop_target[1], 0.6] 
                _, stuck = move_to_position(robot_id, target, obstacles, use_apf=True, strict_orientation=False)

                if stuck: stuck_counter += 1
                else: stuck_counter = max(0, stuck_counter - 1)

                if stuck_counter > 20: 
                    print("⚠️ APF Trapped! Generating True RRT Tree detour...")
                    start_q = [p.getJointState(robot_id, i)[0] for i in range(7)]
                    goal_q = p.calculateInverseKinematics(
                        robot_id, 11, target,
                        lowerLimits=LOWER_LIMITS, upperLimits=UPPER_LIMITS, 
                        jointRanges=JOINT_RANGES, restPoses=REST_POSES,
                        maxNumIterations=200)
                    
                    rrt_planner = RRT(robot_id, obstacles, start_q, goal_q[:7])
                    raw_path = rrt_planner.plan()
                    
                    if raw_path is None:
                        print("❌ RRT FATAL: Could not find path. Shutting down.")
                        state = "IDLE"
                    else:
                        print(f"🌲 RRT Tree found raw path with {len(raw_path)} nodes.")
                        rrt_path = shortcut_path(robot_id, raw_path, obstacles)
                        print(f"✨ Phase B Enhancement: Smoothed path to {len(rrt_path)} nodes.")
                        rrt_path_index = 0
                        stuck_counter = 0

                if get_ee_distance_to_target(robot_id, target) < 0.05:
                    state = "LOWER_TO_PLACE"

        elif state == "LOWER_TO_PLACE":
            target = [active_drop_target[0], active_drop_target[1], active_drop_target[2] + 0.02]
            dist_to_obs = get_closest_obstacle_distance(robot_id, 11, obstacles)
            
            if dist_to_obs < 0.15: 
                print(f"⚠️ Proximity Alert ({dist_to_obs:.2f}m). Applying wrist-twist to clear posture.")
                custom_orientation = p.getQuaternionFromEuler([3.1415, 0, 0.52])
            else:
                custom_orientation = p.getQuaternionFromEuler([3.1415, 0, 0])

            _move_until_reached(robot_id, target, custom_orientation)

            if get_ee_distance_to_target(robot_id, target) < 0.01:
                state = "RELEASE"
                release_timer = time.time()

        elif state == "RELEASE":
            control_gripper(robot_id, open_gripper=True)
            if time.time() - release_timer > 0.5:
                if target_cube in cube_ids:
                    cube_ids.remove(target_cube)
                    cubes_placed += 1 
                state = "POST_RELEASE_LIFT"

        elif state == "POST_RELEASE_LIFT":
            target = [active_drop_target[0], active_drop_target[1], active_drop_target[2] + 0.15]
            move_to_position(robot_id, target, use_apf=False, strict_orientation=True)

            if get_ee_distance_to_target(robot_id, target) < 0.02:
                state = "DONE"

        elif state == "DONE":
            print(f"Total cubes placed: {cubes_placed}")
            if len(cube_ids) > 0:
                print("🌉 Initiating Skybridge return sequence...")
                state = "SAFE_RETRACT_UP" 
            else:
                print("🏁 All cubes placed successfully.")
                state = "IDLE"

        elif state == "SAFE_RETRACT_UP":
            current_ee_pos = p.getLinkState(robot_id, 11)[0]
            target = [current_ee_pos[0], current_ee_pos[1], 0.6] 
            move_to_position(robot_id, target, use_apf=False, strict_orientation=True)
            if get_ee_distance_to_target(robot_id, target) < 0.05:
                state = "SAFE_RETRACT_ACROSS"

        elif state == "SAFE_RETRACT_ACROSS":
            target = [0.3, 0.0, 0.6]
            move_to_position(robot_id, target, use_apf=False, strict_orientation=True)
            if get_ee_distance_to_target(robot_id, target) < 0.05:
                for _ in range(60): p.stepSimulation() 
                target_cube = cube_ids[0]
                state = "VISION_DETECT"

        elif state == "IDLE":
            safe_target = [0.3, 0.0, 0.5]
            move_to_position(robot_id, safe_target, use_apf=False, strict_orientation=True)

        if sim_step_counter % 8 == 0:
            #w_o, h_o, vm, pm, depth, seg, rgb_overhead = get_overhead_camera_data()
            #display_camera_feed("Overhead Camera", rgb_overhead, w_o, h_o)
            w_w, h_w, rgb_wrist, seg_w = get_wrist_camera_data(robot_id)
            display_camera_feed("Wrist Camera", rgb_wrist, w_w, h_w)

        sim_step_counter += 1    
        p.stepSimulation()
        time.sleep(1/240)
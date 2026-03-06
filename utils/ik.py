import numpy as np
import torch
from ikpy.chain import Chain
from scipy.spatial.transform import Rotation as R
from isaaclab.utils.math import subtract_frame_transforms

# Load once

def load_chain():
    chain = Chain.from_urdf_file(
        "robot_urdf/so101_new_calib.urdf",
        # you may need to set these depending on your URDF
        # base_elements=["base_link"],
        # last_link_vector=[0, 0, tcp_offset],
    )

    return chain

def make_T(pos, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
    T[:3, 3] = np.asarray(pos, dtype=np.float64)
    return T

def solve_ik_frame(chain, pos, quat_xyzw, q_init):
    T = make_T(pos, quat_xyzw)
    q_sol = chain.inverse_kinematics_frame(
        T,
        initial_position=q_init,
    )
    return np.asarray(q_sol, dtype=np.float64)

def run_ik_to_pose(env, robot, robot_entity_cfg, ee_jacobi_idx, diff_ik, target_pos_w, target_quat_w, steps=200):
    # target_* must be batched: (num_envs, ...)
    if not torch.is_tensor(target_pos_w):
        target_pos_w = torch.tensor(target_pos_w, dtype=torch.float32, device=env.device)
    if not torch.is_tensor(target_quat_w):
        target_quat_w = torch.tensor(target_quat_w, dtype=torch.float32, device=env.device)

    if target_pos_w.ndim == 1:
        target_pos_w = target_pos_w.unsqueeze(0)
    if target_quat_w.ndim == 1:
        target_quat_w = target_quat_w.unsqueeze(0)

    full_action = None

    for _ in range(steps):
        # current root pose in world
        root_pose_w = robot.data.root_state_w[:, 0:7]
        root_pos_w = root_pose_w[:, 0:3]
        root_quat_w = root_pose_w[:, 3:7]

        # current ee pose in world
        ee_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
        ee_pos_w = ee_pose_w[:, 0:3]
        ee_quat_w = ee_pose_w[:, 3:7]

        # convert current ee pose world -> base frame
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w,
            ee_pos_w, ee_quat_w,
        )

        # convert desired target world -> base frame
        target_pos_b, target_quat_b = subtract_frame_transforms(
            root_pos_w, root_quat_w,
            target_pos_w, target_quat_w,
        )

        jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
        joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]

        print("joint_ids:", robot_entity_cfg.joint_ids)
        print("body_id:", robot_entity_cfg.body_ids[0], "ee_jacobi_idx:", ee_jacobi_idx)
        print("jacobian shape:", jacobian.shape)
        print("pos err:", (target_pos_b - ee_pos_b).detach().cpu().numpy())
        print("quat curr:", ee_quat_b.detach().cpu().numpy())
        print("quat targ:", target_quat_b.detach().cpu().numpy())
        print("joint_pos:", joint_pos.detach().cpu().numpy())

        ik_command = torch.cat([target_pos_b, target_quat_b], dim=-1)
        diff_ik.set_command(ik_command)

        joint_pos_des = diff_ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        print("joint_pos_des:", joint_pos_des.detach().cpu().numpy())
        print("delta:", (joint_pos_des - joint_pos).detach().cpu().numpy())

        if full_action is None:
            # build full action vector
            full_action = robot.data.joint_pos.clone()
            full_action[:, robot_entity_cfg.joint_ids] = joint_pos_des


        env.step(full_action)

        #robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        #env.scene.write_data_to_sim()
        #env.sim.step()
        #env.scene.update(env.sim.get_physics_dt())

def hold_current_joints(env, robot, robot_entity_cfg, steps=1):
    q_hold = robot.data.joint_pos[:, robot_entity_cfg.joint_ids].clone()

    for _ in range(steps):
        robot.set_joint_position_target(q_hold, joint_ids=robot_entity_cfg.joint_ids)
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(env.sim.get_physics_dt())

        q_now = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
        print("joint drift:", (q_now - q_hold).detach().cpu().numpy())
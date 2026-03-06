# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to run a leisaac teleoperation with leisaac manipulation environments."""

"""Launch Isaac Sim Simulator first."""
import multiprocessing

if multiprocessing.get_start_method() != "spawn":
    multiprocessing.set_start_method("spawn", force=True)
import argparse
import signal

from isaaclab.app import AppLauncher


# add argparse arguments
parser = argparse.ArgumentParser(description="leisaac teleoperation for leisaac environments.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    choices=[
        "keyboard",
        "gamepad",
        "so101leader",
        "bi-so101leader",
        "lekiwi-keyboard",
        "lekiwi-gamepad",
        "lekiwi-leader",
    ],
    help="Device for interacting with environment",
)
parser.add_argument(
    "--port", type=str, default="/dev/ttyACM0", help="Port for the teleop device:so101leader, default is /dev/ttyACM0"
)
parser.add_argument(
    "--left_arm_port",
    type=str,
    default="/dev/ttyACM0",
    help="Port for the left teleop device:bi-so101leader, default is /dev/ttyACM0",
)
parser.add_argument(
    "--right_arm_port",
    type=str,
    default="/dev/ttyACM1",
    help="Port for the right teleop device:bi-so101leader, default is /dev/ttyACM1",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed for the environment.")
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")

# recorder_parameter
parser.add_argument("--record", action="store_true", help="whether to enable record function")
parser.add_argument("--step_hz", type=int, default=60, help="Environment stepping rate in Hz.")
parser.add_argument(
    "--dataset_file", type=str, default="./datasets/dataset.hdf5", help="File path to export recorded demos."
)
parser.add_argument("--resume", action="store_true", help="whether to resume recording in the existing dataset file")
parser.add_argument(
    "--num_demos", type=int, default=0, help="Number of demonstrations to record. Set to 0 for infinite."
)

parser.add_argument("--recalibrate", action="store_true", help="recalibrate SO101-Leader or Bi-SO101Leader")
parser.add_argument("--quality", action="store_true", help="whether to enable quality render mode.")
parser.add_argument("--use_lerobot_recorder", action="store_true", help="whether to use lerobot recorder.")
parser.add_argument("--lerobot_dataset_repo_id", type=str, default=None, help="Lerobot Dataset repository ID.")
parser.add_argument("--lerobot_dataset_fps", type=int, default=30, help="Lerobot Dataset frames per second.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

app_launcher_args = vars(args_cli)

# launch omniverse app
app_launcher = AppLauncher(app_launcher_args)
simulation_app = app_launcher.app

import os
import time

import gymnasium as gym
import torch
import numpy as np
from isaaclab.envs import DirectRLEnv, ManagerBasedRLEnv
from isaaclab.managers import DatasetExportMode, TerminationTermCfg
from isaaclab_tasks.utils import parse_env_cfg
from leisaac.enhance.managers import EnhanceDatasetExportMode, StreamingRecorderManager
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim
from utils.point_clouds import sample_mesh_points_global, transform_points, save_pointclouds, rigid_object_pc
from utils.grasps import topdown_antipodal_grasps
from utils.ik import solve_ik_frame, load_chain, run_ik_to_pose, hold_current_joints

from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

from omni.usd import get_context
from pxr import Usd, UsdGeom, Gf


class RateLimiter:
    """Convenience class for enforcing rates in loops."""

    def __init__(self, hz):
        """
        Args:
            hz (int): frequency to enforce
        """
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env):
        """Attempt to sleep at the specified rate in hz."""
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()

        self.last_time = self.last_time + self.sleep_duration

        # detect time jumping forwards (e.g. loop is too slow)
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def manual_terminate(env: ManagerBasedRLEnv | DirectRLEnv, success: bool):
    if hasattr(env, "termination_manager"):
        if success:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.ones(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        else:
            env.termination_manager.set_term_cfg(
                "success",
                TerminationTermCfg(func=lambda env: torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)),
            )
        env.termination_manager.compute()
    elif hasattr(env, "_get_dones"):
        env.cfg.return_success_status = success


def main():  # noqa: C901
    """Running lerobot teleoperation with leisaac manipulation environment."""

    # get directory path and file name (without extension) from cli arguments
    output_dir = os.path.dirname(args_cli.dataset_file)
    output_file_name = os.path.splitext(os.path.basename(args_cli.dataset_file))[0]
    # create directory if it does not exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.use_teleop_device(args_cli.teleop_device)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    task_name = args_cli.task

    if args_cli.quality:
        env_cfg.sim.render.antialiasing_mode = "FXAA"
        env_cfg.sim.render.rendering_mode = "quality"

    # precheck task and teleop device
    if "BiArm" in task_name:
        assert args_cli.teleop_device == "bi-so101leader", "only support bi-so101leader for bi-arm task"
    if "LeKiwi" in task_name:
        assert args_cli.teleop_device in [
            "lekiwi-leader",
            "lekiwi-keyboard",
            "lekiwi-gamepad",
        ], "only support lekiwi-leader, lekiwi-keyboard, lekiwi-gamepad for lekiwi task"
    is_direct_env = "Direct" in task_name
    if is_direct_env:
        assert args_cli.teleop_device in [
            "so101leader",
            "bi-so101leader",
        ], "only support so101leader or bi-so101leader for direct task"

    # timeout and terminate preprocess
    if is_direct_env:
        env_cfg.never_time_out = True
        env_cfg.manual_terminate = True
    else:
        # modify configuration
        if hasattr(env_cfg.terminations, "time_out"):
            env_cfg.terminations.time_out = None
        if hasattr(env_cfg.terminations, "success"):
            env_cfg.terminations.success = None

    # create environment
    env: ManagerBasedRLEnv | DirectRLEnv = gym.make(task_name, cfg=env_cfg).unwrapped

    rate_limiter = RateLimiter(args_cli.step_hz)

    # reset environment
    if hasattr(env, "initialize"):
        env.initialize()
    env.reset()

    interrupted = False

    def signal_handler(signum, frame):
        """Handle SIGINT (Ctrl+C) signal."""
        nonlocal interrupted
        interrupted = True
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Cleaning up resources...")

    original_sigint_handler = signal.signal(signal.SIGINT, signal_handler)

    pcs = []

    chain = load_chain()

    robot = env.scene["robot"]

    print("Robot joint names:", robot.data.joint_names)
    print("Robot body names:", robot.data.body_names)

    robot_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll'],              # better to narrow this to arm joints only
        body_names=['gripper'],          # <-- replace with your actual EE body name
    )
    robot_entity_cfg.resolve(env.scene)

    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
    )
    diff_ik = DifferentialIKController(diff_ik_cfg, num_envs=env.num_envs, device=env.device)

    print("Resolved arm joint ids:", robot_entity_cfg.joint_ids)
    print("Resolved ee body ids:", robot_entity_cfg.body_ids)

    pcs = []

    ee_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
    pre_pos_w = ee_pose_w[:, 0:3].clone()
    grasp_quat_w = ee_pose_w[:, 3:7].clone()

    try:
        while simulation_app.is_running() and not interrupted:
            # run everything in inference mode
            with torch.inference_mode():

                #pc = rigid_object_pc(env, ['Orange001'])
                #pcs.append(pc)

                #grasps = topdown_antipodal_grasps(
                #pc,
                #n_candidates=30,
                #sample_pairs=8000,
                #    min_width=0.015,
                #    max_width=0.080,
                #    approach_clearance=0.08,
                #    finger_depth=0.05,
                #    seed=0,
                #)

                #hold_current_joints(env, robot, robot_entity_cfg, steps=40)

                diff_ik.reset()

                run_ik_to_pose(
                    env, robot, robot_entity_cfg, ee_jacobi_idx, diff_ik,
                    pre_pos_w, grasp_quat_w, steps=40
                )

                grasps = []

                if len(grasps) > 0:
                    g0 = grasps[0]
                    print(
                        f"[GRASP] score={g0['score']:.3f} width={g0['width']:.3f} "
                        f"pos={g0['pos']} quat(xyzw)={g0['quat']}"
                    )

                    # assuming 1 env
                    g = grasps[0]

                    pre_pos_w = torch.tensor(g["pregrasp"], dtype=torch.float32, device=env.device)
                    grasp_pos_w = torch.tensor(g["pos"], dtype=torch.float32, device=env.device)
                    lift_pos_w = grasp_pos_w + torch.tensor([0.0, 0.0, 0.10], dtype=torch.float32, device=env.device)
                    grasp_quat_w = torch.tensor(g["quat"], dtype=torch.float32, device=env.device)


                    diff_ik.reset()

                    #run_ik_to_pose(
                    #    env, robot, robot_entity_cfg, ee_jacobi_idx, diff_ik,
                    #    pre_pos_w, grasp_quat_w, steps=40
                    #)

                    #run_ik_to_pose(
                    #    env, robot, robot_entity_cfg, ee_jacobi_idx, diff_ik,
                    #    grasp_pos_w, grasp_quat_w, steps=40
                    #)

                    # TODO: close gripper here if your action space includes gripper joints

                    #run_ik_to_pose(
                    #    env, robot, robot_entity_cfg, ee_jacobi_idx, diff_ik,
                    #    lift_pos_w, grasp_quat_w, steps=40
                    #)

                if rate_limiter:
                    rate_limiter.sleep(env)
            if interrupted:
                break
    except Exception as e:
        import traceback

        print(f"\n[ERROR] An error occurred: {e}\n")
        traceback.print_exc()
        print("[INFO] Cleaning up resources...")
    finally:

        save_pointclouds(pcs)

        # Restore original signal handler
        signal.signal(signal.SIGINT, original_sigint_handler)
        # finalize the recorder manager
        if args_cli.record and hasattr(env.recorder_manager, "finalize"):
            env.recorder_manager.finalize()
        # close the simulator
        env.close()
        simulation_app.close()


if __name__ == "__main__":
    # example: python teleop.py     --task=LeIsaac-SO101-PickOrange-v0     --teleop_device=so101leader     --port=/dev/ttyACM0     --num_envs=1     --device=cuda     --enable_cameras 
    main()
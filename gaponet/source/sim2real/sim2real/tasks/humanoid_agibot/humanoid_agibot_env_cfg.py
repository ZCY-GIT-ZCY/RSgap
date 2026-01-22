from __future__ import annotations

import os

from sim2real_assets import AGIBOT_G1_CFG, AGIBOT_G1_URDF_PATH  # type: ignore

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass

from .motions.joint_names import ROBOT_JOINT_NAME_DICT_URDF

MOTIONS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "motions")

ROBOT_DICT = {
    "agibot_g1": {
        "model": AGIBOT_G1_CFG,
        "motion_dir": "agibot_g1",
        "urdf_path": AGIBOT_G1_URDF_PATH,
    },
}


@configclass
class HumanoidOperatorEnvCfg(DirectRLEnvCfg):
    robot_name: str = "agibot_g1"
    compute_eq_torque = False

    if "urdf_path" in ROBOT_DICT[robot_name]:
        urdf_model_path = ROBOT_DICT[robot_name]["urdf_path"]
        package_dirs = os.path.dirname(urdf_model_path)
        urdf_joint_name = ROBOT_JOINT_NAME_DICT_URDF[f"{robot_name}_joints"]
    else:
        urdf_model_path = ""
        package_dirs = ""
        urdf_joint_name = ""

    # env
    episode_length_s = 1.0
    decimation = 4

    mode = "train"  # train or play, specified in train.py or play.py

    # spaces
    observation_space = 0
    action_space = 18
    state_space = 0

    early_termination = True
    termination_height = 0.8

    max_payload_mass = 3.0
    robot_mass_range = [1.0, 1.0]

    train_motion_file: str
    reference_body = "base_link"
    reset_strategy = "random"  # default, random, random-start

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**24,
            gpu_total_aggregate_pairs_capacity=2**24,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10.0, replicate_physics=True)

    # robot
    robot: ArticulationCfg = ROBOT_DICT[robot_name]["model"].replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore

    motion_dir = MOTIONS_DIR
    motion_joint = None
    motion_path = os.path.join(motion_dir, f"motion_amass/{ROBOT_DICT[robot_name]['motion_dir']}")
    train_motion_file = os.path.join(motion_path, "motion_agibot.npz")
    test_motion_file = os.path.join(motion_path, "motion_agibot.npz")

    # sub environments
    num_sensor_positions = 1
    sensors_positions = [{}]
    delta_sensor_position = True
    delta_sensor_value = True

    add_model_history = True
    model_history_length = 4  # must match model_history_length in model config
    model_initial_fill_length = 4
    model_history_dim = 54  # 3 * 18 joints

    sensor_dim = 36  # pos + vel for 18 joints
    sensor_decimation = 1

    add_noise = True
    record_sim_mode = False

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

assets_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

AGIBOT_G1_URDF_PATH = os.path.join(assets_dir, "urdfs/agibot_g1/agibot_g1.urdf")

AGIBOT_G1_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=AGIBOT_G1_URDF_PATH,
        activate_contact_sensors=True,
        fix_base=True,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=50.0,
                damping=5.0,
            )
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            ".*_joint": 0.0,
        },
        joint_vel={
            ".*_joint": 0.0,
        },
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "all": ImplicitActuatorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit_sim=100.0,
            velocity_limit_sim=50.0,
            stiffness=50.0,
            damping=5.0,
        ),
    },
)

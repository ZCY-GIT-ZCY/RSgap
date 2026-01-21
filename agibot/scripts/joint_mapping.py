"""
AGIBOT G1 关节映射模块 (Fixed Waist 版本)
=========================================
功能:
1. 定义 observation.state (94维) 和 action (36维) 的索引映射
2. 提供数据提取和转换工具
3. 处理夹爪值域转换

变更说明:
- 腰部关节 (Waist) 已被设为 Fixed Joint，因此从映射中移除。
- 关节维度从 18 维缩减为 16 维。

维度对应关系:
=============
16维关节顺序:
  [0:2]:   头部 (偏航, 俯仰)
  [2:9]:   左臂 (J1-J7)
  [9:16]:  右臂 (J1-J7)
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np


# =============================================================================
# 常量定义
# =============================================================================

# 夹爪转换常量
GRIPPER_STATE_MAX = 120.0   # 真机数据最大值 (闭合)
GRIPPER_URDF_MAX = 0.785    # URDF 关节最大值 (张开, ~45度)


# =============================================================================
# 索引定义
# =============================================================================

@dataclass
class StateIndices:
    """observation.state (94维) 索引定义"""
    # 夹爪
    GRIPPER_L: int = 0
    GRIPPER_R: int = 1
    
    # 关节数据 (注意：不再提取腰部)
    # 原始数据中 [54:68] 包含了 左臂7 + 右臂7
    # 头部在 [82:84]
    
    # 我们将在提取器中显式拼接，不依赖单一连续区间
    
    # 辅助索引
    ARM_L_START: int = 54
    ARM_L_END: int = 61
    ARM_R_START: int = 61
    ARM_R_END: int = 68
    HEAD_START: int = 82
    HEAD_END: int = 84
    
    # 其他未使用的索引保留供参考...
    JOINT_CURRENT: Tuple[int, int] = (68, 82)


@dataclass
class ActionIndices:
    """action (36维) 索引定义"""
    # 夹爪
    GRIPPER_L: int = 0
    GRIPPER_R: int = 1
    
    # 关节目标
    # 原始数据中 [16:30] 包含了 左臂7 + 右臂7
    # 头部在 [30:32]
    
    ARM_L_START: int = 16
    ARM_L_END: int = 23
    ARM_R_START: int = 23
    ARM_R_END: int = 30
    HEAD_START: int = 30
    HEAD_END: int = 32


# =============================================================================
# 关节名称定义
# =============================================================================

# 16维关节顺序 (移除腰部)
JOINT_NAMES_16D = [
    # 头部 (2)
    "idx11_head_joint1",      # 偏航
    "idx12_head_joint2",      # 俯仰
    # 左臂 (7)
    "idx21_arm_l_joint1",
    "idx22_arm_l_joint2",
    "idx23_arm_l_joint3",
    "idx24_arm_l_joint4",
    "idx25_arm_l_joint5",
    "idx26_arm_l_joint6",
    "idx27_arm_l_joint7",
    # 右臂 (7)
    "idx61_arm_r_joint1",
    "idx62_arm_r_joint2",
    "idx63_arm_r_joint3",
    "idx64_arm_r_joint4",
    "idx65_arm_r_joint5",
    "idx66_arm_r_joint6",
    "idx67_arm_r_joint7",
]

# 关节显示名称
JOINT_DISPLAY_NAMES = [
    "head_yaw", "head_pitch",
    "arm_l_j1", "arm_l_j2", "arm_l_j3", "arm_l_j4",
    "arm_l_j5", "arm_l_j6", "arm_l_j7",
    "arm_r_j1", "arm_r_j2", "arm_r_j3", "arm_r_j4",
    "arm_r_j5", "arm_r_j6", "arm_r_j7",
]

# 关节分组 (索引需重新计算)
JOINT_GROUPS = {
    "head": (0, 2),
    "arm_l": (2, 9),
    "arm_r": (9, 16),
}

# 夹爪主驱动关节
GRIPPER_MASTER_JOINTS = {
    "idx41_gripper_l_outer_joint1": "left",
    "idx81_gripper_r_outer_joint1": "right",
}

# 夹爪从动关节 (mimic)
GRIPPER_MIMIC_JOINTS = {
    # ========== 左夹爪 ==========
    "idx31_gripper_l_inner_joint1": ("idx41_gripper_l_outer_joint1", -1.0),
    "idx49_gripper_l_outer_joint2": ("idx41_gripper_l_outer_joint1", 1.5),
    "idx39_gripper_l_inner_joint2": ("idx41_gripper_l_outer_joint1", -1.5),
    "idx42_gripper_l_outer_joint3": ("idx41_gripper_l_outer_joint1", 0.0),
    "idx32_gripper_l_inner_joint3": ("idx41_gripper_l_outer_joint1", 0.0),
    "idx43_gripper_l_outer_joint4": ("idx41_gripper_l_outer_joint1", 0.0),
    "idx33_gripper_l_inner_joint4": ("idx41_gripper_l_outer_joint1", 0.0),
    
    # ========== 右夹爪 ==========
    "idx71_gripper_r_inner_joint1": ("idx81_gripper_r_outer_joint1", -1.0),
    "idx89_gripper_r_outer_joint2": ("idx81_gripper_r_outer_joint1", 1.5),
    "idx79_gripper_r_inner_joint2": ("idx81_gripper_r_outer_joint1", -1.5),
    "idx82_gripper_r_outer_joint3": ("idx81_gripper_r_outer_joint1", 0.0),
    "idx72_gripper_r_inner_joint3": ("idx81_gripper_r_outer_joint1", 0.0),
    "idx83_gripper_r_outer_joint4": ("idx81_gripper_r_outer_joint1", 0.0),
    "idx73_gripper_r_inner_joint4": ("idx81_gripper_r_outer_joint1", 0.0),
}


# 数据提取器
class StateExtractor:
    """从 observation.state (94维) 提取数据"""
    
    idx = StateIndices()
    
    @classmethod
    def extract_joint_positions(cls, state: np.ndarray) -> np.ndarray:
        """
        提取 16 维关节位置
        顺序: 头部(2) + 左臂(7) + 右臂(7)
        """
        head = state[cls.idx.HEAD_START:cls.idx.HEAD_END]
        arm_l = state[cls.idx.ARM_L_START:cls.idx.ARM_L_END]
        arm_r = state[cls.idx.ARM_R_START:cls.idx.ARM_R_END]
        
        return np.concatenate([head, arm_l, arm_r])
    
    @classmethod
    def extract_grippers(cls, state: np.ndarray) -> Tuple[float, float]:
        return float(state[cls.idx.GRIPPER_L]), float(state[cls.idx.GRIPPER_R])


class ActionExtractor:
    """从 action (36维) 提取数据"""
    
    idx = ActionIndices()
    
    @classmethod
    def extract_joint_positions(cls, action: np.ndarray) -> np.ndarray:
        """
        提取 16 维关节目标位置
        顺序: 头部(2) + 左臂(7) + 右臂(7)
        """
        head = action[cls.idx.HEAD_START:cls.idx.HEAD_END]
        arm_l = action[cls.idx.ARM_L_START:cls.idx.ARM_L_END]
        arm_r = action[cls.idx.ARM_R_START:cls.idx.ARM_R_END]
        
        return np.concatenate([head, arm_l, arm_r])
    
    @classmethod
    def extract_grippers(cls, action: np.ndarray) -> Tuple[float, float]:
        return float(action[cls.idx.GRIPPER_L]), float(action[cls.idx.GRIPPER_R])


# =============================================================================
# 夹爪值转换
# =============================================================================

def gripper_state_to_urdf(state_value: float) -> float:
    normalized = np.clip(state_value / GRIPPER_STATE_MAX, 0.0, 1.0)
    return float(GRIPPER_URDF_MAX * (1.0 - normalized))


def gripper_urdf_to_state(urdf_value: float) -> float:
    normalized = 1.0 - np.clip(urdf_value / GRIPPER_URDF_MAX, 0.0, 1.0)
    return float(normalized * GRIPPER_STATE_MAX)


# =============================================================================
# IsaacLab 关节映射器
# =============================================================================

class IsaacLabJointMapper:
    """
    将 16 维关节数据映射到 IsaacLab 关节数组
    """
    
    def __init__(self):
        self.robot_joint_names: List[str] = []
        self.joint_name_to_idx: Dict[str, int] = {}
        self._initialized = False
    
    def set_robot_joint_names(self, joint_names: List[str]):
        self.robot_joint_names = list(joint_names)
        self.joint_name_to_idx = {name: i for i, name in enumerate(joint_names)}
        self._initialized = True
        
        print(f"[JointMapper] 配置 {len(self.robot_joint_names)} 个仿真关节")
    
    def map_16d_to_sim(
        self,
        joint_pos_16d: np.ndarray,
        gripper_l: float,
        gripper_r: float
    ) -> np.ndarray:
        """
        将 16 维关节 + 夹爪映射到仿真数组
        """
        if not self._initialized:
            raise RuntimeError("请先调用 set_robot_joint_names()")
        
        result = np.zeros(len(self.robot_joint_names), dtype=np.float32)
        
        # 1. 映射 16 维关节
        for i, joint_name in enumerate(JOINT_NAMES_16D):
            if joint_name in self.joint_name_to_idx:
                result[self.joint_name_to_idx[joint_name]] = joint_pos_16d[i]
        
        # 2. 映射夹爪主关节
        gripper_l_urdf = gripper_state_to_urdf(gripper_l)
        gripper_r_urdf = gripper_state_to_urdf(gripper_r)
        
        gripper_master_values = {}
        for joint_name, side in GRIPPER_MASTER_JOINTS.items():
            if joint_name in self.joint_name_to_idx:
                value = gripper_l_urdf if side == "left" else gripper_r_urdf
                result[self.joint_name_to_idx[joint_name]] = value
                gripper_master_values[joint_name] = value
        
        # 3. 映射夹爪从动关节
        for slave_name, (master_name, multiplier) in GRIPPER_MIMIC_JOINTS.items():
            if slave_name in self.joint_name_to_idx and master_name in gripper_master_values:
                result[self.joint_name_to_idx[slave_name]] = (
                    gripper_master_values[master_name] * multiplier
                )
        
        return result


# 辅助函数更新
def state_to_16d(state: np.ndarray) -> np.ndarray:
    return StateExtractor.extract_joint_positions(state)

def action_to_16d(action: np.ndarray) -> np.ndarray:
    return ActionExtractor.extract_joint_positions(action)

def get_joint_names() -> List[str]:
    return JOINT_NAMES_16D.copy()

def get_joint_display_names() -> List[str]:
    return JOINT_DISPLAY_NAMES.copy()
"""
AGIBOT G1 关节映射配置 - 修正版 v8

修复：夹爪数据转换 (state [0,120] -> URDF [0.785, 0])
"""

from typing import Dict, List, Tuple
import numpy as np


class G1JointMapping:
    """
    G1机器人关节映射
    
    夹爪数据转换：
    =============
    真机数据: state[0]/state[1] 范围 [0, 120]
      - 0 = 夹爪完全张开
      - 120 = 夹爪完全闭合
    
    URDF 定义:     
    - idx41_outer_joint1: limit [0, 0.785], 主驱动
    - idx31_inner_joint1: limit [-0.785, 0], mimic=-1
    0 = 夹爪闭合
    0.785 = outer右手夹爪张开（inner左夹爪是-0.785）
    
    转换公式: urdf = 0.785 * (1 - state / 120)
    """
    
    # 夹爪转换常量
    GRIPPER_STATE_MAX = 120.0   # 真机数据最大值
    GRIPPER_URDF_MAX = 0.785    # URDF 关节最大值
    
    def __init__(self, joint_names: List[str] = None):
        if joint_names is not None:
            self.robot_joint_names = list(joint_names)
        else:
            self.robot_joint_names = []
        
        self._build_mappings()
    
    def _build_mappings(self):
        """构建映射表"""
        
        # 1. 直接映射关节 (不含夹爪主关节)
        self.direct_map: Dict[str, Tuple[int, float]] = {
            # 腰部
            "idx01_body_joint1": (84, 10),
            "idx02_body_joint2": (85, 1.0),
            # 头部
            "idx11_head_joint1": (82, 1.0),
            "idx12_head_joint2": (83, 1.0),
            # 左臂 (state[54:61])
            "idx21_arm_l_joint1": (54, 1.0),
            "idx22_arm_l_joint2": (55, 1.0),
            "idx23_arm_l_joint3": (56, 1.0),
            "idx24_arm_l_joint4": (57, 1.0),
            "idx25_arm_l_joint5": (58, 1.0),
            "idx26_arm_l_joint6": (59, 1.0),
            "idx27_arm_l_joint7": (60, 1.0),
            # 右臂 (state[61:68])
            "idx61_arm_r_joint1": (61, 1.0),
            "idx62_arm_r_joint2": (62, 1.0),
            "idx63_arm_r_joint3": (63, 1.0),
            "idx64_arm_r_joint4": (64, 1.0),
            "idx65_arm_r_joint5": (65, 1.0),
            "idx66_arm_r_joint6": (66, 1.0),
            "idx67_arm_r_joint7": (67, 1.0),
        }
        
        # 2. 夹爪主驱动关节
        self.gripper_master_joints: Dict[str, Tuple[int, float]] = {
            "idx41_gripper_l_outer_joint1": (0, 1.0),
            "idx81_gripper_r_outer_joint1": (1, 1.0), 
        }
        
        # 3. 从动关节 (mimic)
        self.mimic_map: Dict[str, Tuple[str, float]] = {

            # ========== 左夹爪 ==========
            "idx31_gripper_l_inner_joint1": ("idx41_gripper_l_outer_joint1", -1.0),
            "idx49_gripper_l_outer_joint2": ("idx41_gripper_l_outer_joint1", 1.5),
            "idx39_gripper_l_inner_joint2": ("idx41_gripper_l_outer_joint1", -1.5),
            "idx42_gripper_l_outer_joint3": ("idx41_gripper_l_outer_joint1", 0),
            "idx32_gripper_l_inner_joint3": ("idx41_gripper_l_outer_joint1", 0),
            "idx43_gripper_l_outer_joint4": ("idx41_gripper_l_outer_joint1", 0),
            "idx33_gripper_l_inner_joint4": ("idx41_gripper_l_outer_joint1", 0),
            
            # ========== 右夹爪 ==========
            # r_outer_joint 1 是主关节，系数1.0
            "idx71_gripper_r_inner_joint1": ("idx81_gripper_r_outer_joint1", -1.0),
            "idx89_gripper_r_outer_joint2": ("idx81_gripper_r_outer_joint1", 1.5), # 控制右手夹爪外侧的内连杆
            "idx79_gripper_r_inner_joint2": ("idx81_gripper_r_outer_joint1", -1.5),
            "idx82_gripper_r_outer_joint3": ("idx81_gripper_r_outer_joint1", 0), # 控制右手夹爪外侧的外子连杆向外打开程度，0→1
            "idx72_gripper_r_inner_joint3": ("idx81_gripper_r_outer_joint1", 0),
            "idx83_gripper_r_outer_joint4": ("idx81_gripper_r_outer_joint1", 0), # 控制右手夹爪外侧的外末端连杆（关节）向外打开程度，0→1
            "idx73_gripper_r_inner_joint4": ("idx81_gripper_r_outer_joint1", 0),
        }
    
    def set_robot_joint_names(self, joint_names: List[str]):
        """设置 IsaacLab 返回的实际关节顺序"""
        self.robot_joint_names = list(joint_names)
        print(f"[Mapping] 已配置 {len(self.robot_joint_names)} 个关节")
    
    def _convert_gripper_value(self, state_value: float) -> float:
        """
        转换夹爪值: state [0, 120] -> URDF [0.785, 0]
        
        state=0 (张开) -> urdf=0.785
        state=120 (闭合) -> urdf=0
        """
        normalized = np.clip(state_value / self.GRIPPER_STATE_MAX, 0.0, 1.0)
        urdf_value = self.GRIPPER_URDF_MAX * (1.0 - normalized)
        return urdf_value
    
    def state_to_isaaclab_array(self, state: np.ndarray) -> np.ndarray:
        """将 observation.state (94维) 转换为 IsaacLab 关节目标数组"""
        if not self.robot_joint_names:
            raise ValueError("请先调用 set_robot_joint_names()")
        
        result = np.zeros(len(self.robot_joint_names), dtype=np.float32)
        
        # 预计算夹爪主关节的 URDF 值
        gripper_urdf_values: Dict[str, float] = {}
        for joint_name, (state_idx, multiplier) in self.gripper_master_joints.items():
            base_value = self._convert_gripper_value(state[state_idx])
            gripper_urdf_values[joint_name] = base_value * multiplier
        
        for i, joint_name in enumerate(self.robot_joint_names):
            if joint_name in self.direct_map:
                # 直接映射 (手臂、头部、腰部)
                state_idx, multiplier = self.direct_map[joint_name]
                result[i] = state[state_idx] * multiplier
            
            elif joint_name in self.gripper_master_joints:
                # 夹爪主驱动关节
                result[i] = gripper_urdf_values[joint_name]
            
            elif joint_name in self.mimic_map:
                # 从动关节
                master_name, multiplier = self.mimic_map[joint_name]
                
                if master_name in gripper_urdf_values:
                    # Mimic 夹爪关节
                    result[i] = gripper_urdf_values[master_name] * multiplier
                elif master_name in self.direct_map:
                    # Mimic 其他关节 (目前没有)
                    master_idx, master_mul = self.direct_map[master_name]
                    result[i] = state[master_idx] * master_mul * multiplier
        
        return result
    
    def print_debug(self, state: np.ndarray):
        """打印调试信息"""
        print("\n" + "="*70)
        print("[Debug] 关节映射详情")
        print("="*70)
        
        # 夹爪数据
        print(f"\n[夹爪数据转换]:")
        for joint_name, (state_idx, multiplier) in self.gripper_master_joints.items():
            raw = state[state_idx]
            base_value = self._convert_gripper_value(raw)
            final_value = base_value * multiplier
            print(f"  {joint_name}:")
            print(f"    state[{state_idx}] = {raw:.4f} (0=张开, 120=闭合)")
            print(f"    -> URDF = {base_value:.4f} * {multiplier:+.1f} = {final_value:+.4f}")
        
        # 全部关节
        print(f"\n[映射结果]:")
        result = self.state_to_isaaclab_array(state)
        for i, (name, val) in enumerate(zip(self.robot_joint_names, result)):
            if "gripper" in name:
                print(f"  [{i:2d}] {name:<35} = {val:+.4f} (夹爪)")
            else:
                print(f"  [{i:2d}] {name:<35} = {val:+.4f}")
        
        print("="*70)
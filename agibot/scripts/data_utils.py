"""
AGIBOT G1 数据工具模块
======================
功能:
1. 加载 parquet 源数据和 info.json 配置
2. 解析 observation.state (94维) 和 action (36维) 的字段映射
3. 时间戳对齐与线性插值
4. 数据预处理与异常处理

数据维度说明 (基于 info.json):
- observation.state (94维):
  - [0]: 左夹爪位置 [0,120]
  - [1]: 右夹爪位置 [0,120]
  - [2:14]: 末端力/力矩 (12维)
  - [14:20]: 末端位置 (6维)
  - [20:32]: 末端速度 (12维)
  - [32:40]: 末端姿态四元数 (8维)
  - [40:48]: 手臂基座姿态 (8维)
  - [48:54]: 手臂基座位置 (6维)
  - [54:68]: 关节位置 (14维: 左臂7+右臂7) ★ 核心
  - [68:82]: 关节电流 (14维)
  - [82:84]: 头部位置 (2维)
  - [84:86]: 腰部位置 (2维)
  - [86:89]: 底盘位置 (3维) - 忽略
  - [89:93]: 底盘姿态 (4维) - 忽略
  - [93]: 控制源状态 - 忽略

- action (36维):
  - [0]: 左夹爪目标 [0,120]
  - [1]: 右夹爪目标 [0,120]
  - [2:8]: 末端目标位置 (6维)
  - [8:16]: 末端目标姿态 (8维)
  - [16:30]: 关节目标位置 (14维) 核心
  - [30:32]: 头部目标 (2维)
  - [32:34]: 腰部目标 (2维)
  - [34:36]: 底盘速度 (2维) - 忽略
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import warnings


# =============================================================================
# 数据结构定义
# =============================================================================

@dataclass
class FieldDescriptor:
    """字段描述符"""
    name: str
    start_idx: int
    end_idx: int
    dimensions: int
    description: str = ""
    
    def extract(self, data: np.ndarray) -> np.ndarray:
        """从数据中提取该字段"""
        return data[..., self.start_idx:self.end_idx]


@dataclass
class DatasetConfig:
    """数据集配置 (从 info.json 解析)"""
    robot_type: str
    fps: float
    total_episodes: int
    total_frames: int
    
    # 字段映射
    state_fields: Dict[str, FieldDescriptor] = field(default_factory=dict)
    action_fields: Dict[str, FieldDescriptor] = field(default_factory=dict)
    
    @property
    def dt(self) -> float:
        """采样时间间隔"""
        return 1.0 / self.fps


@dataclass
class FrameRecord:
    """单帧完整记录"""
    timestamp: float
    frame_index: int

    # 原始数据
    observation_state: np.ndarray  # 94维
    action: np.ndarray             # 36维

    # 解析后的关节数据 (SI 单位: 弧度)
    # 注意: 腰部关节已固定，不再包含在关节数据中
    real_joint_pos: np.ndarray     # 18维: 头部2+左臂7+右臂7+夹爪2
    real_joint_vel: Optional[np.ndarray] = None  # 速度 (如果可用)
    target_joint_pos: np.ndarray = None  # 18维: 来自 action
    
    # 夹爪数据 (原始值 [0,120])
    real_gripper_l: float = 0.0
    real_gripper_r: float = 0.0
    target_gripper_l: float = 0.0
    target_gripper_r: float = 0.0

    # 末端执行器位置 (6维: 左手xyz + 右手xyz)
    real_end_effector_pos: Optional[np.ndarray] = None
    target_end_effector_pos: Optional[np.ndarray] = None


@dataclass
class AlignedData:
    """对齐后的仿真与真实数据"""
    timestamps: np.ndarray         # (N,) 统一时间戳

    # 真实数据
    real_joint_pos: np.ndarray     # (N, num_joints)
    target_joint_pos: np.ndarray   # (N, num_joints)

    # 仿真数据
    sim_joint_pos: np.ndarray      # (N, num_joints)
    sim_joint_vel: np.ndarray      # (N, num_joints)
    sim_joint_torque: np.ndarray   # (N, num_joints)

    # 夹爪数据
    real_gripper_l: Optional[np.ndarray] = None      # (N,) 左夹爪真实值
    real_gripper_r: Optional[np.ndarray] = None      # (N,) 右夹爪真实值
    target_gripper_l: Optional[np.ndarray] = None    # (N,) 左夹爪目标值
    target_gripper_r: Optional[np.ndarray] = None    # (N,) 右夹爪目标值
    sim_gripper_l: Optional[np.ndarray] = None       # (N,) 左夹爪仿真值
    sim_gripper_r: Optional[np.ndarray] = None       # (N,) 右夹爪仿真值

    # 末端执行器位置 (6维: 左手xyz + 右手xyz)
    real_end_effector_pos: Optional[np.ndarray] = None    # (N, 6)
    target_end_effector_pos: Optional[np.ndarray] = None  # (N, 6)

    # 元信息
    joint_names: List[str] = field(default_factory=list)
    source_fps: float = 30.0


# =============================================================================
# info.json 解析器
# =============================================================================

class InfoJsonParser:
    """
    解析 info.json 中的字段描述
    
    info.json 格式示例:
    {
        "fps": 30,
        "features": {
            "observation.state": {
                "shape": [94],
                "field_descriptions": {
                    "state/joint/position": {"dimensions": 14, ...},
                    ...
                }
            },
            "action": {
                "shape": [36],
                "field_descriptions": {...}
            }
        }
    }
    """
    
    # 预定义的索引映射 (基于分析结果)
    STATE_INDICES = {
        "left_effector_position": (0, 1),
        "right_effector_position": (1, 2),
        "end_wrench": (2, 14),
        "end_position": (14, 20),
        "end_velocity": (20, 32),
        "end_orientation": (32, 40),
        "arm_orientation": (40, 48),
        "arm_position": (48, 54),
        "joint_position": (54, 68),      # 14维关节位置
        "joint_current": (68, 82),
        "head_position": (82, 84),       # 头部
        "waist_position": (84, 86),      # 腰部
        "robot_position": (86, 89),      # 忽略
        "robot_orientation": (89, 93),   # 忽略
        "action_src_status": (93, 94),   # 忽略
    }
    
    ACTION_INDICES = {
        "left_effector_position": (0, 1),
        "right_effector_position": (1, 2),
        "end_position": (2, 8),
        "end_orientation": (8, 16),
        "joint_position": (16, 30),      # ★ 14维关节位置
        "head_position": (30, 32),       # ★ 头部
        "waist_position": (32, 34),      # ★ 腰部
        "robot_velocity": (34, 36),      # 忽略
    }
    
    def __init__(self, info_path: Path):
        self.info_path = Path(info_path)
        self.raw_config = self._load_json()
        
    def _load_json(self) -> dict:
        with open(self.info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def parse(self) -> DatasetConfig:
        """解析配置并返回 DatasetConfig"""
        config = DatasetConfig(
            robot_type=self.raw_config.get("robot_type", "unknown"),
            fps=float(self.raw_config.get("fps", 30)),
            total_episodes=self.raw_config.get("total_episodes", 0),
            total_frames=self.raw_config.get("total_frames", 0),
        )
        
        # 构建字段描述符
        for name, (start, end) in self.STATE_INDICES.items():
            config.state_fields[name] = FieldDescriptor(
                name=name,
                start_idx=start,
                end_idx=end,
                dimensions=end - start,
            )
        
        for name, (start, end) in self.ACTION_INDICES.items():
            config.action_fields[name] = FieldDescriptor(
                name=name,
                start_idx=start,
                end_idx=end,
                dimensions=end - start,
            )
        
        return config


# =============================================================================
# 数据加载器
# =============================================================================

class DataLoader:
    """
    AGIBOT 数据加载器
    
    支持:
    - 加载 parquet 格式的 episode 数据
    - 解析字段并转换为标准格式
    - 处理时间戳抖动和缺失值
    """
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.meta_path = self.dataset_path / "meta"
        self.data_path = self.dataset_path / "data"
        
        # 加载配置
        info_path = self.meta_path / "info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"info.json 不存在: {info_path}")
        
        parser = InfoJsonParser(info_path)
        self.config = parser.parse()
        
        print(f"[DataLoader] 数据集: {self.dataset_path.name}")
        print(f"[DataLoader] 机器人类型: {self.config.robot_type}")
        print(f"[DataLoader] FPS: {self.config.fps}")
        print(f"[DataLoader] 总帧数: {self.config.total_frames}")
    
    def load_episode(self, episode_index: int) -> List[FrameRecord]:
        """
        加载单个 episode 的所有帧
        
        Args:
            episode_index: Episode 编号
            
        Returns:
            List[FrameRecord]: 帧记录列表
        """
        # 定位 parquet 文件
        chunk_size = 1000
        chunk_index = episode_index // chunk_size
        parquet_path = (
            self.data_path 
            / f"chunk-{chunk_index:03d}" 
            / f"episode_{episode_index:06d}.parquet"
        )
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Episode 文件不存在: {parquet_path}")
        
        # 读取数据
        df = pd.read_parquet(parquet_path)
        
        frames = []
        for _, row in df.iterrows():
            obs_state = np.array(row["observation.state"], dtype=np.float32)
            action = np.array(row["action"], dtype=np.float32)
            
            real_joint_pos = self._extract_full_joint_positions(obs_state)
            target_joint_pos = self._extract_full_joint_positions_from_action(action)

            # 提取末端执行器位置 (6维: 左手xyz + 右手xyz)
            real_end_effector = obs_state[14:20]  # observation.state[14:20]
            target_end_effector = action[2:8]     # action[2:8]

            frame = FrameRecord(
                timestamp=float(row["timestamp"]),
                frame_index=int(row["frame_index"]),
                observation_state=obs_state,
                action=action,
                real_joint_pos=real_joint_pos,
                target_joint_pos=target_joint_pos,
                real_gripper_l=obs_state[0],
                real_gripper_r=obs_state[1],
                target_gripper_l=action[0],
                target_gripper_r=action[1],
                real_end_effector_pos=real_end_effector,
                target_end_effector_pos=target_end_effector,
            )
            frames.append(frame)
        
        print(f"[DataLoader] Episode {episode_index}: 加载 {len(frames)} 帧")
        return frames
    
    def _extract_full_joint_positions(self, state: np.ndarray) -> np.ndarray:
        """
        从 observation.state 提取完整关节位置 (18维)

        顺序:
        - [0:2]: 头部 (state[82:84])
        - [2:9]: 左臂 (state[54:61])
        - [9:16]: 右臂 (state[61:68])
        - [16:18]: 夹爪 (state[0], state[1])
        """
        head = state[82:84]    # 2维
        arm_l = state[54:61]   # 7维
        arm_r = state[61:68]   # 7维
        grip_l = self._convert_gripper_raw_to_joint(state[0])
        grip_r = self._convert_gripper_raw_to_joint(state[1])

        return np.concatenate([head, arm_l, arm_r, [grip_l, grip_r]])
    
    def _extract_full_joint_positions_from_action(self, action: np.ndarray) -> np.ndarray:
        """
        从 action 提取目标关节位置 (18维)

        顺序:
        - [0:2]: 头部 (action[30:32])
        - [2:9]: 左臂 (action[16:23])
        - [9:16]: 右臂 (action[23:30])
        - [16:18]: 夹爪 (action[0], action[1])

        """
        head = action[30:32]   # 2维
        arm_l = action[16:23]  # 7维
        arm_r = action[23:30]  # 7维
        grip_l = self._convert_gripper_raw_to_joint(action[0])
        grip_r = self._convert_gripper_raw_to_joint(action[1])

        return np.concatenate([head, arm_l, arm_r, [grip_l, grip_r]])

    def _convert_gripper_raw_to_joint(self, value: float) -> float:
        """
        Convert raw gripper value [0, 120] into joint position (rad).
        Uses the URDF joint limits for outer_joint1: [0, pi/4].
        """
        raw = float(value)
        raw_clamped = min(max(raw, JointNameMapper.GRIPPER_RAW_MIN), JointNameMapper.GRIPPER_RAW_MAX)
        if raw != raw_clamped:
            warnings.warn(f"Gripper value {raw} out of range, clamped to {raw_clamped}")
        ratio = (raw_clamped - JointNameMapper.GRIPPER_RAW_MIN) / (
            JointNameMapper.GRIPPER_RAW_MAX - JointNameMapper.GRIPPER_RAW_MIN
        )
        return float(
            JointNameMapper.GRIPPER_JOINT_MIN
            + ratio * (JointNameMapper.GRIPPER_JOINT_MAX - JointNameMapper.GRIPPER_JOINT_MIN)
        )
    
    def get_episode_count(self) -> int:
        """获取 episode 总数"""
        return self.config.total_episodes


# =============================================================================
# 时间对齐工具
# =============================================================================

class TimeAligner:
    """
    时间对齐与线性插值工具
    
    功能:
    1. 处理时间戳抖动 (dt 不完全等于 1/fps)
    2. 线性插值对齐仿真数据到真实数据时间网格
    3. 处理 NaN 和异常值
    """
    
    @staticmethod
    def validate_timestamps(timestamps: np.ndarray, expected_dt: float, tolerance: float = 0.5) -> bool:
        """
        验证时间戳有效性
        
        Args:
            timestamps: 时间戳数组
            expected_dt: 期望的时间间隔
            tolerance: 容差比例 (0.5 表示允许 50% 偏差)
            
        Returns:
            bool: 是否有效
        """
        if len(timestamps) < 2:
            return True
        
        dts = np.diff(timestamps)
        min_dt = expected_dt * (1 - tolerance)
        max_dt = expected_dt * (1 + tolerance)
        
        outliers = np.sum((dts < min_dt) | (dts > max_dt))
        if outliers > 0:
            warnings.warn(
                f"发现 {outliers} 个异常时间间隔 "
                f"(期望 {expected_dt:.4f}s, 范围 [{min_dt:.4f}, {max_dt:.4f}])"
            )
        
        return outliers < len(dts) * 0.1  # 允许最多 10% 异常
    
    @staticmethod
    def interpolate_to_target_timestamps(
        source_timestamps: np.ndarray,
        source_data: np.ndarray,
        target_timestamps: np.ndarray,
        fill_value: str = "extrapolate"
    ) -> np.ndarray:
        """
        将源数据插值到目标时间戳
        
        Args:
            source_timestamps: 源时间戳 (N_src,)
            source_data: 源数据 (N_src, D) 或 (N_src,)
            target_timestamps: 目标时间戳 (N_tgt,)
            fill_value: 边界处理方式
            
        Returns:
            插值后的数据 (N_tgt, D) 或 (N_tgt,)
        """
        # 处理 1D 情况
        if source_data.ndim == 1:
            return np.interp(target_timestamps, source_timestamps, source_data)
        
        # 处理 2D 情况
        result = np.zeros((len(target_timestamps), source_data.shape[1]), dtype=np.float32)
        for i in range(source_data.shape[1]):
            result[:, i] = np.interp(
                target_timestamps, 
                source_timestamps, 
                source_data[:, i]
            )
        
        return result
    
    @staticmethod
    def handle_nan_values(data: np.ndarray, method: str = "interpolate") -> np.ndarray:
        """
        处理 NaN 值
        
        Args:
            data: 输入数据
            method: 处理方法 ("interpolate", "zero", "forward_fill")
            
        Returns:
            处理后的数据
        """
        if not np.any(np.isnan(data)):
            return data
        
        result = data.copy()
        
        if method == "zero":
            result[np.isnan(result)] = 0.0
            
        elif method == "forward_fill":
            # 前向填充
            if result.ndim == 1:
                mask = np.isnan(result)
                idx = np.where(~mask, np.arange(len(result)), 0)
                np.maximum.accumulate(idx, out=idx)
                result = result[idx]
            else:
                for i in range(result.shape[1]):
                    mask = np.isnan(result[:, i])
                    idx = np.where(~mask, np.arange(len(result)), 0)
                    np.maximum.accumulate(idx, out=idx)
                    result[:, i] = result[idx, i]
                    
        elif method == "interpolate":
            # 线性插值填充
            if result.ndim == 1:
                mask = ~np.isnan(result)
                if np.sum(mask) >= 2:
                    result = np.interp(
                        np.arange(len(result)),
                        np.arange(len(result))[mask],
                        result[mask]
                    )
            else:
                for i in range(result.shape[1]):
                    mask = ~np.isnan(result[:, i])
                    if np.sum(mask) >= 2:
                        result[:, i] = np.interp(
                            np.arange(len(result)),
                            np.arange(len(result))[mask],
                            result[mask, i]
                        )
        
        return result


# =============================================================================
# 关节名称映射
# =============================================================================

class JointNameMapper:
    """
    关节名称映射器
    
    将数据索引映射到 URDF 关节名称
    """
    
    # 16维关节顺序 (与数据提取顺序一致，腰部已固定)
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

    # 分组索引 (16维)
    HEAD_INDICES = (0, 2)
    ARM_L_INDICES = (2, 9)
    ARM_R_INDICES = (9, 16)

    # 夹爪原始范围 [0, 120] 映射到 URDF 关节范围 [0, pi/4]
    GRIPPER_RAW_MIN = 0.0
    GRIPPER_RAW_MAX = 120.0
    GRIPPER_JOINT_MIN = 0.0
    GRIPPER_JOINT_MAX = np.pi / 4.0

    # 18维关节顺序 (加入夹爪)
    JOINT_NAMES_18D = JOINT_NAMES_16D + [
        "idx41_gripper_l_outer_joint1",
        "idx81_gripper_r_outer_joint1",
    ]

    @classmethod
    def get_joint_names(cls, include_gripper: bool = True) -> List[str]:
        if include_gripper:
            return cls.JOINT_NAMES_18D.copy()
        return cls.JOINT_NAMES_16D.copy()
    
    @classmethod
    def get_group_indices(cls, group: str) -> Tuple[int, int]:
        """获取关节组的索引范围 (18维)"""
        groups = {
            "head": cls.HEAD_INDICES,
            "arm_l": cls.ARM_L_INDICES,
            "arm_r": cls.ARM_R_INDICES,
            "gripper": (16, 18),
        }
        return groups.get(group, (0, 18))


# =============================================================================
# 导出函数
# =============================================================================

def load_episode_data(dataset_path: str, episode_index: int) -> Tuple[List[FrameRecord], DatasetConfig]:
    """
    便捷函数: 加载 episode 数据
    
    Returns:
        (帧列表, 配置)
    """
    loader = DataLoader(dataset_path)
    frames = loader.load_episode(episode_index)
    return frames, loader.config


def frames_to_arrays(frames: List[FrameRecord]) -> Dict[str, np.ndarray]:
    """
    将帧列表转换为 numpy 数组

    Returns:
        {
            "timestamps": (N,),
            "real_joint_pos": (N, 18),
            "target_joint_pos": (N, 18),
            "real_gripper_l": (N,),
            "real_gripper_r": (N,),
            "real_end_effector_pos": (N, 6),
            "target_end_effector_pos": (N, 6),
            ...
        }
    """
    n = len(frames)

    result = {
        "timestamps": np.array([f.timestamp for f in frames]),
        "frame_indices": np.array([f.frame_index for f in frames]),
        "real_joint_pos": np.stack([f.real_joint_pos for f in frames]),
        "target_joint_pos": np.stack([f.target_joint_pos for f in frames]),
        "real_gripper_l": np.array([f.real_gripper_l for f in frames]),
        "real_gripper_r": np.array([f.real_gripper_r for f in frames]),
        "target_gripper_l": np.array([f.target_gripper_l for f in frames]),
        "target_gripper_r": np.array([f.target_gripper_r for f in frames]),
    }

    # 添加末端执行器位置 (如果存在)
    if frames[0].real_end_effector_pos is not None:
        result["real_end_effector_pos"] = np.stack([f.real_end_effector_pos for f in frames])
    if frames[0].target_end_effector_pos is not None:
        result["target_end_effector_pos"] = np.stack([f.target_end_effector_pos for f in frames])

    return result


if __name__ == "__main__":
    # 测试代码
    import sys
    
    dataset_path = "/home/jianing/Desktop/yuntian/agibot/data/H3_example"
    
    loader = DataLoader(dataset_path)
    frames = loader.load_episode(0)
    
    print(f"\n加载 {len(frames)} 帧")
    
    # 转换为数组
    arrays = frames_to_arrays(frames)
    
    print(f"\n数据维度:")
    for k, v in arrays.items():
        print(f"  {k}: {v.shape}")
    
    # 验证时间戳
    aligner = TimeAligner()
    valid = aligner.validate_timestamps(arrays["timestamps"], 1.0/30.0)
    print(f"\n时间戳有效性: {valid}")
    
    # 打印关节名称
    print(f"\n关节名称 18维):")
    for i, name in enumerate(JointNameMapper.get_joint_names()):
        print(f"  [{i:2d}] {name}")
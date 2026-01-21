"""
AGIBOT G1motion 数据加载模块
用于解析真机采集的parquet数据和meta信息
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    raise ImportError("请安装依赖: pip install pandas pyarrow")


@dataclass
class DatasetInfo:
    """数据集元信息"""
    robot_type: str
    total_episodes: int
    total_frames: int
    fps: float
    data_path_template: str
    video_path_template: str
    features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeInfo:
    """单个Episode信息"""
    episode_index: int
    tasks: List[str]
    length: int  # 帧数


@dataclass 
class FrameData:
    """单帧数据"""
    timestamp: float
    frame_index: int
    observation_state: np.ndarray  # 94维
    action: np.ndarray  # 36维
    
    # 解析后的关节数据
    joint_positions: Optional[np.ndarray] = None  # 14维 (左臂7+右臂7)
    head_positions: Optional[np.ndarray] = None   # 2维
    waist_positions: Optional[np.ndarray] = None  # 2维
    left_gripper: Optional[float] = None
    right_gripper: Optional[float] = None


class AgibotDataLoader:
    """
    AGIBOT数据集加载器
    
    用法:
        loader = AgibotDataLoader("/path/to/H3_example")
        loader.load_meta()
        episode_data = loader.load_episode(0)
    """
    
    # observation.state 字段索引定义
    STATE_INDICES = {
        "left_effector_position": 0,
        "right_effector_position": 1,
        "end_wrench": (2, 14),           # 12维: 左右末端力/力矩
        "end_position": (14, 20),        # 6维: 左右末端位置
        "end_velocity": (20, 32),        # 12维
        "end_orientation": (32, 40),     # 8维: 左右末端四元数
        "arm_orientation": (40, 48),     # 8维
        "arm_position": (48, 54),        # 6维
        "joint_position": (54, 68),      # 14维: 关节角度
        "joint_current": (68, 82),       # 14维: 关节电流
        "head_position": (82, 84),       # 2维: 头部
        "waist_position": (84, 86),      # 2维: 腰部
        "robot_position": (86, 89),      # 3维: 底盘位置
        "robot_orientation": (89, 93),   # 4维: 底盘四元数
        "action_src_status": 93,         # 1维
    }
    
    # action 字段索引定义
    ACTION_INDICES = {
        "left_effector_position": 0,
        "right_effector_position": 1,
        "end_position": (2, 8),          # 6维
        "end_orientation": (8, 16),      # 8维
        "joint_position": (16, 30),      # 14维: 关节目标角度
        "head_position": (30, 32),       # 2维
        "waist_position": (32, 34),      # 2维
        "robot_velocity": (34, 36),      # 2维
    }
    
    def __init__(self, dataset_path: str):
        """
        初始化数据加载器
        
        Args:
            dataset_path: 数据集根目录路径 (如 /path/to/H3_example)
        """
        self.dataset_path = Path(dataset_path)
        self.meta_path = self.dataset_path / "meta"
        self.data_path = self.dataset_path / "data"
        
        self.info: Optional[DatasetInfo] = None
        self.episodes: List[EpisodeInfo] = []
        self.tasks: Dict[int, str] = {}
        
    def load_meta(self) -> None:
        """加载所有元数据文件"""
        self._load_info()
        self._load_episodes()
        self._load_tasks()
        print(f"[DataLoader] 加载完成: {self.info.total_episodes} episodes, "
              f"{self.info.total_frames} frames @ {self.info.fps} FPS")
    
    def _load_info(self) -> None:
        """加载info.json"""
        info_path = self.meta_path / "info.json"
        with open(info_path, "r") as f:
            data = json.load(f)
        
        self.info = DatasetInfo(
            robot_type=data.get("robot_type", "unknown"),
            total_episodes=data.get("total_episodes", 0),
            total_frames=data.get("total_frames", 0),
            fps=data.get("fps", 30.0),
            data_path_template=data.get("data_path", ""),
            video_path_template=data.get("video_path", ""),
            features=data.get("features", {}),
        )
    
    def _load_episodes(self) -> None:
        """加载episodes.jsonl"""
        episodes_path = self.meta_path / "episodes.jsonl"
        self.episodes = []
        
        with open(episodes_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                self.episodes.append(EpisodeInfo(
                    episode_index=data["episode_index"],
                    tasks=data["tasks"],
                    length=data["length"],
                ))
    
    def _load_tasks(self) -> None:
        """加载tasks.jsonl"""
        tasks_path = self.meta_path / "tasks.jsonl"
        self.tasks = {}
        
        with open(tasks_path, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                self.tasks[data["task_index"]] = data["task"]
    
    def get_episode_info(self, episode_index: int) -> EpisodeInfo:
        """获取指定episode的信息"""
        return self.episodes[episode_index]
    
    def load_episode(self, episode_index: int) -> List[FrameData]:
        """
        加载单个episode的所有帧数据
        
        Args:
            episode_index: Episode序号
            
        Returns:
            List[FrameData]: 该episode的所有帧数据列表
        """
        # 确定chunk编号
        chunk_size = 1000
        chunk_index = episode_index // chunk_size
        
        # 构建文件路径
        parquet_path = self.data_path / f"chunk-{chunk_index:03d}" / f"episode_{episode_index:06d}.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Episode文件不存在: {parquet_path}")
        
        # 读取parquet文件
        df = pd.read_parquet(parquet_path)
        
        frames = []
        for _, row in df.iterrows():
            obs_state = np.array(row["observation.state"], dtype=np.float32)
            action = np.array(row["action"], dtype=np.float32)
            
            frame = FrameData(
                timestamp=float(row["timestamp"]),
                frame_index=int(row["frame_index"]),
                observation_state=obs_state,
                action=action,
            )
            
            # 解析关键字段
            frame.joint_positions = self._extract_joint_positions(obs_state)
            frame.head_positions = self._extract_head_positions(obs_state)
            frame.waist_positions = self._extract_waist_positions(obs_state)
            frame.left_gripper = obs_state[0]
            frame.right_gripper = obs_state[1]
            
            frames.append(frame)
        
        return frames
    
    def _extract_joint_positions(self, state: np.ndarray) -> np.ndarray:
        """提取14维关节角度 (左臂7 + 右臂7)"""
        start, end = self.STATE_INDICES["joint_position"]
        return state[start:end]
    
    def _extract_head_positions(self, state: np.ndarray) -> np.ndarray:
        """提取头部位置 (2维)"""
        start, end = self.STATE_INDICES["head_position"]
        return state[start:end]
    
    def _extract_waist_positions(self, state: np.ndarray) -> np.ndarray:
        """提取腰部位置 (2维)"""
        start, end = self.STATE_INDICES["waist_position"]
        return state[start:end]
    
    def get_joint_trajectory(self, episode_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取整个episode的关节轨迹
        
        Args:
            episode_index: Episode序号
            
        Returns:
            timestamps: (N,) 时间戳数组
            joint_positions: (N, 14) 关节角度数组
        """
        frames = self.load_episode(episode_index)
        
        timestamps = np.array([f.timestamp for f in frames])
        joint_positions = np.array([f.joint_positions for f in frames])
        
        return timestamps, joint_positions


def print_data_summary(loader: AgibotDataLoader, episode_index: int = 0):
    """打印数据摘要信息"""
    print("\n" + "="*60)
    print("AGIBOT G1motion 数据集摘要")
    print("="*60)
    
    print(f"\n📁 数据集路径: {loader.dataset_path}")
    print(f"🤖 机器人类型: {loader.info.robot_type}")
    print(f"📊 总Episodes: {loader.info.total_episodes}")
    print(f"🎞️  总帧数: {loader.info.total_frames}")
    print(f"⏱️  采集帧率: {loader.info.fps} FPS")
    
    # 加载一个episode进行展示
    frames = loader.load_episode(episode_index)
    print(f"\n📌 Episode {episode_index} 详情:")
    print(f"   帧数: {len(frames)}")
    print(f"   时长: {frames[-1].timestamp:.2f} 秒")
    
    # 打印第一帧数据维度
    first_frame = frames[0]
    print(f"\n📐 数据维度:")
    print(f"   observation.state: {first_frame.observation_state.shape}")
    print(f"   action: {first_frame.action.shape}")
    print(f"   joint_positions: {first_frame.joint_positions.shape}")
    
    # 打印关节角度范围
    timestamps, joint_traj = loader.get_joint_trajectory(episode_index)
    print(f"\n🔧 关节角度统计 (弧度):")
    print(f"   最小值: {joint_traj.min(axis=0)}")
    print(f"   最大值: {joint_traj.max(axis=0)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import sys
    
    # 默认数据集路径
    default_path = "/home/jianing/Desktop/yuntian/agibot/Bus_table_example/H3_example"
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    # 加载数据
    loader = AgibotDataLoader(dataset_path)
    loader.load_meta()
    
    # 打印摘要
    print_data_summary(loader)

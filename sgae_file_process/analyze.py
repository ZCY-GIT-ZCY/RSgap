import argparse
import numpy as np
import os
from pathlib import Path

def analyze_npz_files(root_dir: str):
    """
    递归遍历指定目录，分析所有 motor_all_joints.npz 文件的结构。

    Args:
        root_dir (str): 要开始搜索的根目录。
    """
    print(f"������ Starting analysis in root directory: {root_dir}\n")
    
    # 使用 pathlib.Path.rglob 来递归查找所有匹配的文件
    # npz_files = sorted(list(Path(root_dir).rglob("motor_all_joints.npz")))
    npz_files = sorted(list(Path(root_dir).rglob("merged_all.npz")))
    if not npz_files:
        print(f"❌ No 'motor_all_joints.npz' files found in '{root_dir}'.")
        print("Please check if the directory is correct and if sagegr3_process.py has been run successfully.")
        return

    print(f"✅ Found {len(npz_files)} NPZ files to analyze.\n")

    # 存储第一个文件的结构，用于后续比较
    first_file_structure = None
    inconsistent_files = []

    for i, file_path in enumerate(npz_files):
        print(f"--- [{i+1}/{len(npz_files)}] Analyzing: {file_path} ---")
        
        try:
            # allow_pickle=True 是必需的，因为 joint_sequence 是对象数组
            with np.load(file_path, allow_pickle=True) as data:
                keys = list(data.keys())
                print(f"  Keys found: {keys}")

                current_structure = {}

                for key in keys:
                    array = data[key]
                    shape = array.shape
                    dtype = array.dtype
                    print(f"  - Key: '{key}'")
                    print(f"    - Shape: {shape}")
                    print(f"    - Dtype: {dtype}")
                    
                    current_structure[key] = (shape, str(dtype))

                    # 特别显示 joint_sequence 的内容
                    if key == 'joint_sequence':
                        num_joints = len(array)
                        print(f"    - Number of joints: {num_joints}")
                        print(f"    - Joint names: {list(array)}")
                        # 更新结构信息，包含关节数量
                        current_structure[key] = (shape, str(dtype), num_joints)

                # 与第一个文件的结构进行比较
                if first_file_structure is None:
                    first_file_structure = current_structure
                elif current_structure != first_file_structure:
                    inconsistent_files.append(file_path)

        except Exception as e:
            print(f"  ������ Error loading or analyzing file: {e}")
        
        print("-" * (len(str(file_path)) + 20))
        print() # 添加空行以分隔

    # 打印最终摘要
    print("\n--- Analysis Summary ---")
    if first_file_structure:
        print("Structure of the first analyzed file (used as a reference):")
        for key, (shape, dtype, *rest) in first_file_structure.items():
            if key == 'joint_sequence':
                print(f"  - '{key}': Shape={shape}, Dtype={dtype}, NumJoints={rest[0]}")
            else:
                print(f"  - '{key}': Shape={shape}, Dtype={dtype}")
    
    if inconsistent_files:
        print(f"\n������ Warning: {len(inconsistent_files)} files had a structure inconsistent with the first file:")
        for f_path in inconsistent_files:
            print(f"  - {f_path}")
    else:
        print("\n✅ All analyzed files have a consistent data structure.")


def parse_args():
    p = argparse.ArgumentParser(description="Analyze the data structure of generated NPZ files.")
    p.add_argument(
        "--root", 
        type=str, 
        default="output_files/gr3v2_2_2_real/first/real_3kg_1", 
        help="The root directory containing the NPZ files to analyze."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyze_npz_files(root_dir=args.root)
#!/usr/bin/env python3
"""
测试关节限位读取功能
验证修复后的 get_joint_limits() 方法是否能正确读取URDF中的限位
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from scripts.sim_runner import IsaacLabSimRunner, SimConfig

def test_urdf_parsing():
    """测试URDF解析功能"""
    print("=" * 80)
    print("测试1: 直接从URDF文件解析关节限位")
    print("=" * 80)

    # 创建配置
    config = SimConfig(
        urdf_path="assets/G1_omnipicker/urdf/G1_omnipicker.urdf"
    )

    # 创建runner（不初始化仿真）
    runner = IsaacLabSimRunner(config)

    # 直接测试URDF解析方法
    joint_limits = runner._parse_urdf_joint_limits()

    if not joint_limits:
        print("❌ 失败: 无法从URDF解析关节限位")
        return False

    print(f"✅ 成功解析 {len(joint_limits)} 个关节限位\n")

    # 检查关键关节
    critical_joints = {
        'idx01_body_joint1': {'expected_lower': 0.0, 'expected_upper': 0.55, 'type': 'prismatic'},
        'idx02_body_joint2': {'expected_lower': 0.0, 'expected_upper': 1.5708, 'type': 'revolute'},
    }

    print("检查关键关节限位:")
    all_passed = True

    for joint_name, expected in critical_joints.items():
        if joint_name in joint_limits:
            limit = joint_limits[joint_name]
            lower = limit['lower']
            upper = limit['upper']
            joint_type = limit.get('type', 'unknown')

            # 检查限位是否正确
            lower_ok = abs(lower - expected['expected_lower']) < 0.01
            upper_ok = abs(upper - expected['expected_upper']) < 0.01
            type_ok = joint_type == expected['type']

            if lower_ok and upper_ok and type_ok:
                print(f"  ✅ {joint_name} ({joint_type}): [{lower:.4f}, {upper:.4f}]")
            else:
                print(f"  ❌ {joint_name} ({joint_type}): [{lower:.4f}, {upper:.4f}]")
                print(f"     期望: [{expected['expected_lower']:.4f}, {expected['expected_upper']:.4f}]")
                all_passed = False
        else:
            print(f"  ❌ {joint_name}: 未找到")
            all_passed = False

    return all_passed

def main():
    print("\n" + "=" * 80)
    print("关节限位修复验证测试")
    print("=" * 80 + "\n")

    # 测试URDF解析
    test1_passed = test_urdf_parsing()

    print("\n" + "=" * 80)
    print("测试总结")
    print("=" * 80)

    if test1_passed:
        print("✅ 所有测试通过！")
        print("\n修复说明:")
        print("1. URDF文件中的关节限位定义正常")
        print("2. 新增了 _parse_urdf_joint_limits() 方法作为备用方案")
        print("3. get_joint_limits() 现在有3种读取方式:")
        print("   - 方法1: 从 soft_joint_pos_limits 读取")
        print("   - 方法2: 从 root_physx_view 读取")
        print("   - 方法3: 直接从URDF文件解析（备用）")
        print("\n建议:")
        print("- 运行完整的仿真测试: python scripts/replay_action.py --episode 0")
        print("- 检查输出中的关节限位信息是否正确显示")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())

"""
修复 URDF 文件以适配 IsaacLab
1. 将 package:// 路径转换为绝对路径
2. 移除 loop_joint 标签
"""

import re
from pathlib import Path


def fix_urdf_for_isaaclab(
    urdf_path: str,
    mesh_base_dir: str,
    output_path: str = None
) -> str:
    """
    修复 URDF 文件
    
    Args:
        urdf_path: 原始 URDF 文件路径
        mesh_base_dir: mesh 文件所在的基础目录
        output_path: 输出文件路径（可选）
    
    Returns:
        修复后的文件路径
    """
    urdf_path = Path(urdf_path)
    mesh_base = Path(mesh_base_dir)
    
    if output_path is None:
        output_path = urdf_path.parent / f"{urdf_path.stem}_isaaclab.urdf"
    else:
        output_path = Path(output_path)
    
    with open(urdf_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # ========== 1. 修复 mesh 路径 ==========
    # 匹配 package://genie_robot_description/meshes/xxx
    def replace_mesh_path(match):
        relative_path = match.group(1)  # e.g., "G1/base_link.fbx"
        abs_path = mesh_base / relative_path
        return f'filename="{abs_path}"'
    
    # 替换 package:// 格式的路径
    pattern = r'filename="package://genie_robot_description/meshes/([^"]+)"'
    content = re.sub(pattern, replace_mesh_path, content)
    
    # ========== 2. 移除 loop_joint 标签 ==========
    # 匹配整个 loop_joint 元素（包括多行）
    loop_pattern = r'<loop_joint[^>]*>.*?</loop_joint>\s*'
    content = re.sub(loop_pattern, '', content, flags=re.DOTALL)
    
    # 也移除自闭合的 loop_joint（以防万一）
    content = re.sub(r'<loop_joint[^/]*/>\s*', '', content)
    
    # ========== 3. 移除相关注释 ==========
    content = re.sub(r'<!--\s*Gripper Finger Close-loop Structure\s*-->\s*', '', content)
    
    # ========== 保存修复后的文件 ==========
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✅ URDF 修复完成！")
    print(f"   输入: {urdf_path}")
    print(f"   输出: {output_path}")
    
    # 统计修改
    with open(urdf_path, 'r') as f:
        original = f.read()
    
    original_package_count = original.count('package://')
    fixed_package_count = content.count('package://')
    original_loop_count = original.count('<loop_joint')
    fixed_loop_count = content.count('<loop_joint')
    
    print(f"\n修改统计:")
    print(f"   package:// 路径: {original_package_count} → {fixed_package_count}")
    print(f"   loop_joint 标签: {original_loop_count} → {fixed_loop_count}")
    
    return str(output_path)


def verify_urdf(urdf_path: str) -> bool:
    """验证 URDF 文件"""
    import xml.etree.ElementTree as ET
    
    print(f"\n验证 URDF: {urdf_path}")
    print("-" * 50)
    
    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        print(f"✅ XML 解析成功")
        print(f"   Robot name: {root.attrib.get('name', 'N/A')}")
    except ET.ParseError as e:
        print(f"❌ XML 解析失败: {e}")
        return False
    
    # 统计关节
    joints = root.findall('.//joint')
    revolute = [j for j in joints if j.get('type') == 'revolute']
    prismatic = [j for j in joints if j.get('type') == 'prismatic']
    fixed = [j for j in joints if j.get('type') == 'fixed']
    
    print(f"\n关节统计:")
    print(f"   Revolute: {len(revolute)}")
    print(f"   Prismatic: {len(prismatic)}")
    print(f"   Fixed: {len(fixed)}")
    print(f"   总计: {len(joints)}")
    
    # 检查 mesh 路径
    meshes = root.findall('.//mesh')
    package_paths = []
    missing_files = []
    
    for mesh in meshes:
        filename = mesh.get('filename', '')
        if filename.startswith('package://'):
            package_paths.append(filename)
        elif not Path(filename).exists():
            missing_files.append(filename)
    
    if package_paths:
        print(f"\n⚠️  发现 {len(package_paths)} 个未修复的 package:// 路径")
        for p in package_paths[:3]:
            print(f"   - {p}")
        if len(package_paths) > 3:
            print(f"   ... 还有 {len(package_paths) - 3} 个")
    
    if missing_files:
        print(f"\n⚠️  发现 {len(missing_files)} 个缺失的 mesh 文件")
        for f in missing_files[:3]:
            print(f"   - {f}")
    
    # 检查 loop_joint
    loop_joints = root.findall('.//loop_joint')
    if loop_joints:
        print(f"\n⚠️  发现 {len(loop_joints)} 个 loop_joint（IsaacLab 不支持）")
    
    success = not package_paths and not loop_joints
    if success:
        print(f"\n✅ URDF 验证通过，可用于 IsaacLab")
    else:
        print(f"\n❌ URDF 需要进一步修复")
    
    return success


if __name__ == "__main__":
    # 配置路径
    PROJECT_ROOT = Path("/home/jianing/Desktop/yuntian/agibot")
    
    URDF_INPUT = PROJECT_ROOT / "assets/G1_omnipicker/urdf/G1_omnipicker.urdf"
    MESH_DIR = PROJECT_ROOT / "assets/G1_omnipicker/meshes"
    URDF_OUTPUT = PROJECT_ROOT / "assets/G1_omnipicker/urdf/G1_omnipicker_isaaclab.urdf"
    
    # 检查 mesh 目录
    if not MESH_DIR.exists():
        print(f"⚠️  Mesh 目录不存在: {MESH_DIR}")
        print("请先复制 mesh 文件到该目录")
        print("\n建议执行:")
        print(f"  mkdir -p {MESH_DIR}")
        print(f"  cp -r /path/to/genie_robot_description/meshes/* {MESH_DIR}/")
    else:
        # 修复 URDF
        fixed_path = fix_urdf_for_isaaclab(
            str(URDF_INPUT),
            str(MESH_DIR),
            str(URDF_OUTPUT)
        )
        
        # 验证修复后的 URDF
        verify_urdf(fixed_path)
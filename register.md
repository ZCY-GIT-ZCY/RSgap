# Agibot 接入 GapONet 操作清单

以下是把 `agibot` 的 `G1_omnipicker` 接入 `gaponet` 的完整步骤清单（不包含 npz 与关节顺序细节）。

---

## 1) 资产文件准备（URDF + meshes）

### 1.1 复制文件
- 复制 URDF：
  - 源：`agibot/assets/G1_omnipicker/urdf/G1_omnipicker.urdf`
  - 目标：`gaponet/source/sim2real_assets/urdfs/agibot_g1/agibot_g1.urdf`
- 复制 meshes（保持子目录结构）：
  - 源：`agibot/assets/G1_omnipicker/meshes/`
  - 目标：`gaponet/source/sim2real_assets/urdfs/agibot_g1/meshes/`
  - 目录要求：
    - `meshes/G1/*.fbx`
    - `meshes/omnipicker/*.dae`

### 1.2 修正 URDF mesh 路径
把 URDF 中的 mesh 路径从：
```
package://genie_robot_description/meshes/xxx.fbx
```
改成：
```
meshes/G1/xxx.fbx
```
以及（夹爪相关 dae）：
```
meshes/omnipicker/xxx.dae
```
说明：URDF 使用 `sim2real_assets/urdfs/<robot>/meshes` 下的子目录结构。

---

## 2) 新增机器人配置（ArticulationCfg）

### 2.1 新建机器人配置文件
新增文件：`gaponet/source/sim2real_assets/sim2real_assets/robots/agibot_g1.py`

内容要点：
- 使用 `UrdfFileCfg`（URDF 直接导入，IsaacLab 会自动转 USD）。
- 指定 `urdf_path` 为 `sim2real_assets/urdfs/agibot_g1/agibot_g1.urdf`。
- 合理填写 `init_state`（初始关节位置）。
- 设置 `actuators`（可先按默认阻尼/刚度起步）。
- 导出常量：
  - `AGIBOT_G1_CFG`
  - `AGIBOT_G1_URDF_PATH`

### 2.2 注册 robots 包
修改文件：`gaponet/source/sim2real_assets/sim2real_assets/robots/__init__.py`
- 添加：`from .agibot_g1 import *`

---

## 3) 新增关节/链接名称字典（匹配 motion 数据）

修改文件：`gaponet/source/sim2real/sim2real/tasks/humanoid_agibot/motions/joint_names.py`

新增条目：
- `ROBOT_BODY_JOINT_NAME_DICT["agibot_g1_joints"] = [...]`
- `ROBOT_BODY_JOINT_NAME_DICT["agibot_g1_links"] = [...]`
- `ROBOT_JOINT_NAME_DICT_URDF["agibot_g1_joints"] = [...]`

说明：
- joints 顺序必须与 `motion_agibot.npz` 的 `joint_sequence` 一致。
- links 顺序需与 URDF 中链接名称一致，用于 body index/对齐。
- 当前版本按 **16 关节**（头 2 + 左臂 7 + 右臂 7）。

---

## 3.5) motion 数据放置位置（README 口径 + 现有代码路径）

README 只要求：
- “Place in appropriate motion directory”

结合新的 `humanoid_agibot` 目录结构，建议放到：
- `gaponet/source/sim2real/sim2real/tasks/humanoid_agibot/motions/motion_amass/agibot_g1/motion_agibot.npz`

随后在环境配置中：
- `motion_dir = "agibot_g1"`
- 或直接将 `train_motion_file` / `test_motion_file` 指向该 `.npz`

---

## 4) 新增环境配置（避免影响现有 H1）

### 4.0 建立独立任务目录（与 README 一致）
- 新建目录：`gaponet/source/sim2real/sim2real/tasks/humanoid_agibot/`
- 参考 `humanoid_operator` 结构复制，并做如下迁移：
  - `humanoid_operator_env.py` → `humanoid_agibot_env.py`
  - `humanoid_operator_agibot_env_cfg.py` → `humanoid_agibot_env_cfg.py`
  - 旧目录中 `humanoid_operator_agibot_env_cfg.py` 删除，避免误用
- 训练脚本导入任务模块：
  - `gaponet/scripts/reinforcement_learning/rsl_rl/train.py` 增加 `import sim2real.tasks.humanoid_agibot`
  - `gaponet/scripts/reinforcement_learning/rl_games/train.py` 增加 `import sim2real.tasks.humanoid_agibot`

### 4.1 新建 agibot 环境配置
新增文件：
`gaponet/source/sim2real/sim2real/tasks/humanoid_agibot/humanoid_agibot_env_cfg.py`

修改点（基于现有 `humanoid_operator_env_cfg.py` 复制并迁移到独立目录）：
- 增加 `ROBOT_DICT` 条目：
  ```
  "agibot_g1": {
      "model": AGIBOT_G1_CFG,
      "motion_dir": "agibot_g1",
      "urdf_path": AGIBOT_G1_URDF_PATH
  }
  ```
- `robot_name = "agibot_g1"`
- `train_motion_file` / `test_motion_file` 指向 agibot 的 motion 目录
- `reference_body` 改成 URDF 中实际存在的主干 link（如 `base_link` 或 `body_link1`）
- 根据 `joint_sequence` 长度同步：
  - `action_space`
  - `sensor_dim`
  - `model_history_dim`（如果启用历史）

### 4.1.1 运行期修正（payload 与 URDF）
为避免 URDF 中缺失 payload prim 导致报错，做了如下修正：

- 机器人配置 `agibot_g1.py`
  - `UrdfFileCfg` 使用 `asset_path=...`（不是 `urdf_path`）
  - `asset_path` 指向 `gaponet/source/sim2real_assets/urdfs/agibot_g1/agibot_g1.urdf`
  - 增加 `joint_drive.gains`（`stiffness=50.0`, `damping=5.0`）
- 环境 `humanoid_agibot_env.py`
  - payload prim 放到 `/World/envs/env_.*/payload{1..4}`（不挂在 `Robot/` 下面）
  - 为 payload 创建小球占位（`SphereCfg`，`radius=0.02`，`kinematic_enabled=True`，`disable_gravity=True`）

说明：这些 payload 仅用于承载质量随机化逻辑；若后续把 payload 写进 URDF，可移除小球占位。

### 4.1.2 运行期修正（motion npz 读取与兼容）
为解决 IsaacSim/Kit 在读取 object dtype 的 `npz` 时卡住/报错，做了如下修正：

- `agibot/scripts/convert_parquet_to_npz.py`
  - 新增 dense padded keys：
    - `real_dof_positions_padded`
    - `real_dof_velocities_padded`
    - `real_dof_positions_cmd_padded`
    - `real_dof_torques_padded`
    - `motion_len`
  - `joint_sequence` / `joint_names` 保存为 **字符串数组**（`dtype=str`），避免 object pickle 问题
- `sim2real/tasks/humanoid_agibot/motions/motion_motor_loader.py`
  - 优先读取 dense padded keys（若存在）
  - 读取 `joint_sequence` / `joint_names` 时强制转为 `dtype=str`
  - `wrist_index` 在无腕关节时返回空张量，避免硬编码腕关节报错

### 4.1.3 运行期修正（运动关节映射）
为解决机器人关节数（34）与 motion DOF（16）不一致导致的 shape mismatch：
- `humanoid_agibot_env.py` 中新增 motion→robot 关节索引映射
- 所有写入/读取机器人关节状态与 target 的操作使用该映射（`joint_ids`）

### 4.1.4 运行期修正（policy 观测格式）
为兼容 RSL-RL runner（要求 `obs` 为 tensor 而非 dict）：
- `humanoid_agibot_env.py` 中 `policy` 观测改为 `branch+trunk` 拼接后的 **单一张量**
- 同时保留 `model` 与 `operator` 字段用于调试/记录

### 4.1.5 运行期修正（DeepONet 类注册）
为解决 `DeepONetActorCritic` 在 rsl-rl 里 `eval()` 找不到的问题：
- `humanoid_agibot/agents/rsl_rl_operator_cfg.py` 增加 `from sim2real.rsl_rl.modules import DeepONetActorCritic`
- `scripts/reinforcement_learning/rsl_rl/train.py` 注入 `DeepONetActorCritic` 到 `rsl_rl.runners.on_policy_runner` 的全局命名空间

### 4.2 注册新环境任务
修改文件：`gaponet/source/sim2real/sim2real/tasks/humanoid_agibot/__init__.py`

新增注册（示例）：
```
gym.register(
    id="Isaac-Humanoid-AGIBOT-Delta-Action",
    entry_point=f"{__name__}.humanoid_agibot_env:HumanoidOperatorEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_agibot_env_cfg:HumanoidOperatorEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_operator_cfg:HumanoidOperatorRunnerCfg",
    },
)
```

---

## 5) 资产加载方式确认

当前建议：**URDF 直接加载**（不强制 USD）。

如后续需要 USD：
- 生成 USD 并放入 `gaponet/source/sim2real_assets/usds/agibot_g1/`
- 修改 `agibot_g1.py` 中的 `spawn` 从 `UrdfFileCfg` 切换成 `UsdFileCfg`

---

## 6) 训练/评估入口

使用新 task id：
```
python scripts/rsl_rl/train.py --task Isaac-Humanoid-AGIBOT-Delta-Action ...
```

或直接覆盖 env 参数：
```
python scripts/rsl_rl/train.py --task Isaac-Humanoid-AGIBOT-Delta-Action \
  env.robot_name=agibot_g1 \
  env.train_motion_file=... \
  env.test_motion_file=...
```

---

## 7) 需要你提供/确认的配置项

已按“上身固定”的口径先取一组可用初值，后续可调整：

- URDF：`fix_base=True`（上身固定）
- 主要根 link（`reference_body`）：`base_link`
- 关节驱动参数（`ImplicitActuatorCfg`）：
  - `effort_limit_sim=100.0`
  - `velocity_limit_sim=50.0`
  - `stiffness=50.0`
  - `damping=5.0`
  - `soft_joint_pos_limit_factor=0.9`
- 自碰撞：`enabled_self_collisions=False`

**正确性风险提示（agent 输入维度）**  
`critic_input_dim` 中 `robot_mass` 的长度按 42 links 估算（来自 agibot link 列表）。  
若实际 `num_bodies` 不为 42，需要改为：  
`sensor_dim*num_sensor_positions + 16 + 48 + 32 + 1 + 2 + num_bodies`

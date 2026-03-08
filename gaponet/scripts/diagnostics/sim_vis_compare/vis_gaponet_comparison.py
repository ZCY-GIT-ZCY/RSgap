"""
GAPONet 效果对比可视化脚本

显示 3 个机器人并排（Y轴排列）：
- REAL           : 真实机器人轨迹 (珊瑚红，半透明)
- SIM without GAPONet (青白色)
- SIM with GAPONet    (金黄色)

可以直观对比 GAPONet 是否减小了 sim-real gap。

需要先运行 precompute_delta_actions.py 生成预计算数据。

Usage:
    # Step 1: 预计算 delta actions
    python precompute_delta_actions.py \
        --task Isaac-SO101-Operator-Delta-Action \
        --model /path/to/model.pt \
        --motion_file /path/to/test.npz \
        --motion_idx 0 \
        --output_file /path/to/precomputed.npz
    
    # Step 2: 可视化
    python vis_gaponet_comparison.py \
        --robot-name so101 \
        --precomputed-file /path/to/precomputed.npz
"""

import argparse
import ctypes
import os
from datetime import datetime

import cv2
import numpy as np


def log_message(message):
    """Format and print log messages with timestamp."""
    import sys
    current_time = datetime.now().strftime("%H:%M:%S")
    msg = f"[VisGAPO][Log][{current_time}] {message}\n"
    sys.stderr.write(msg)
    sys.stderr.flush()


class GAPONetComparisonVisualizer:
    """
    Visualize GAPONet comparison with 3 robots in a row:
    - REAL           (coral-red, semi-transparent ghost)
    - SIM without GAPONet  (icy cyan-white)
    - SIM with GAPONet     (vivid gold)

    Uses precomputed delta actions from precompute_delta_actions.py
    """

    def __init__(self, args, sim_module):
        self.args = args
        self.sim_module = sim_module
        
        self.robot_name = args.robot_name.lower()
        self.precomputed_file = args.precomputed_file
        self.group_offset = args.group_offset
        self.headless = args.headless
        self.physics_freq = args.physics_freq
        self.render_freq = args.render_freq
        self.control_freq = args.control_freq
        self.save_video = args.save_video
        self.video_path = args.video_path
        self.video_fps = args.video_fps
        self.video_writer = None
        self.viewport_api = None
        
        # Check frequency divisibility
        if self.physics_freq % self.render_freq != 0:
            raise ValueError("Physics frequency must be divisible by render frequency.")
        if self.render_freq % self.control_freq != 0:
            raise ValueError("Render frequency must be divisible by control frequency.")

        # Initialize simulation parameters
        self.physics_dt = 1.0 / self.physics_freq
        self.render_dt = 1.0 / self.render_freq
        self.control_dt = 1.0 / self.control_freq
        self.divisor = self.render_freq // self.control_freq
        
        # Import robot config
        from assets import get_robot_config
        self.robot_config = get_robot_config(self.robot_name)
        
        # Get PD gains (try both key variants used in assets.py)
        self.joint_kp = self.robot_config.get_config_value("joint_kp", None)
        if self.joint_kp is None:
            self.joint_kp = self.robot_config.get_config_value("default_kp", None)
        self.joint_kd = self.robot_config.get_config_value("joint_kd", None)
        if self.joint_kd is None:
            self.joint_kd = self.robot_config.get_config_value("default_kd", None)
        
        # Load precomputed data first so we know num_dofs
        self._load_precomputed_data()

        # Now expand scalar gains to per-joint arrays
        if self.joint_kp is None:
            self.joint_kp = [100.0] * self.num_dofs
        elif not isinstance(self.joint_kp, list):
            self.joint_kp = [float(self.joint_kp)] * self.num_dofs
        if self.joint_kd is None:
            self.joint_kd = [10.0] * self.num_dofs
        elif not isinstance(self.joint_kd, list):
            self.joint_kd = [float(self.joint_kd)] * self.num_dofs
        log_message(f"Using PD gains: Kp={self.joint_kp}, Kd={self.joint_kd}")
        
        # Setup simulation with 3 robots
        self._setup_simulation()
    
    def _load_precomputed_data(self):
        """Load precomputed delta actions and motion data."""
        log_message(f"Loading precomputed data from {self.precomputed_file}...")
        
        if not os.path.exists(self.precomputed_file):
            raise FileNotFoundError(f"Precomputed file not found: {self.precomputed_file}")
        
        data = np.load(self.precomputed_file, allow_pickle=True)
        
        self.motion_idx = int(data['motion_idx'])
        self.num_frames = int(data['num_frames'])
        self.num_dofs = int(data['num_dofs'])
        
        self.commands = np.array(data['commands'])
        self.delta_actions = np.array(data['delta_actions'])
        self.real_positions = np.array(data['real_positions'])

        # Preferred joint names from precompute file
        if 'dof_names' in data:
            self.joint_names = [str(x) for x in data['dof_names'].tolist()]
        elif 'joint_names' in data:
            self.joint_names = [str(x) for x in data['joint_names'].tolist()]
        else:
            # Fallback: derive names from robot_name + num_dofs
            if self.robot_name == "agibot" and self.num_dofs == 14:
                self.joint_names = [
                    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
                    "arm_l_joint5", "arm_l_joint6", "arm_l_joint7",
                    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
                    "arm_r_joint5", "arm_r_joint6", "arm_r_joint7",
                ]
            else:
                # Legacy SO101 default
                self.joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        
        log_message(f"Motion index: {self.motion_idx}")
        log_message(f"Loaded {self.num_frames} frames with {self.num_dofs} DOFs")
        log_message(f"Joint names ({len(self.joint_names)}): {self.joint_names}")
        log_message(f"Delta action range: [{self.delta_actions.min():.4f}, {self.delta_actions.max():.4f}] rad")
        log_message(f"Mean |delta|: {np.abs(self.delta_actions).mean():.4f} rad ({np.rad2deg(np.abs(self.delta_actions).mean()):.2f}°)")
    
    def _setup_simulation(self):
        """Set up simulation with 3 robots."""
        import omni
        
        sim_module = self.sim_module
        
        # Robot paths (3 overlapping + 1 standalone REAL on the right)
        self.prim_paths = {
            'sim_nogap':   "/World/Robot_SIM_noGAP",
            'real_1':      "/World/Robot_REAL",
            'sim_withgap': "/World/Robot_SIM_withGAP",
            'real_2':      "/World/Robot_REAL_2",
        }
        
        self.robot_usd_path = self.robot_config.usd_path
        # Check if robot uses URDF instead of USD
        self.robot_urdf_path = self.robot_config.get_config_value("urdf_path", None)
        base_offset = list(self.robot_config.offset)
        
        # Robot layout: 3 robots overlapping at center + 1 standalone REAL to the right.
        # "right" = +Y from camera perspective (camera at +X looking toward -X).
        side_offset = self.group_offset  # default 1.8 m to the right
        positions = {
            'sim_nogap':   list(base_offset),
            'real_1':      list(base_offset),
            'sim_withgap': list(base_offset),
            'real_2':      [base_offset[0], base_offset[1] + side_offset, base_offset[2]],
        }
        
        # Create stage
        sim_module.create_new_stage()
        stage = omni.usd.get_context().get_stage()
        self.stage = stage
        
        # Simulation config
        sim_params = {
            "gravity": [0, 0, -9.81],
            "solver_type": 1,  # TGS
        }
        
        # Initialize world
        self.world = sim_module.World(
            physics_dt=self.physics_dt,
            rendering_dt=self.render_dt,
            stage_units_in_meters=1.0,
            sim_params=sim_params
        )
        
        # Add ground plane
        self.world.scene.add_default_ground_plane()
        # Paint the ground plane black
        try:
            from pxr import UsdShade, Sdf, Gf as _Gf2
            import omni.usd as _ousd2
            _s2 = _ousd2.get_context().get_stage()
            _floor_paths = [
                "/World/defaultGroundPlane/CollisionMesh",
                "/World/defaultGroundPlane/CollisionPlane",
                "/World/defaultGroundPlane",
            ]
            _floor_mat = UsdShade.Material.Define(_s2, "/World/FloorBlackMat")
            _floor_sh  = UsdShade.Shader.Define(_s2, "/World/FloorBlackMat/Shader")
            _floor_sh.CreateIdAttr("UsdPreviewSurface")
            _floor_sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(_Gf2.Vec3f(0.0, 0.0, 0.0))
            _floor_sh.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(1.0)
            _floor_sh.CreateInput("metallic",     Sdf.ValueTypeNames.Float).Set(0.0)
            _floor_mat.CreateSurfaceOutput().ConnectToSource(_floor_sh.ConnectableAPI(), "surface")
            for _fp in _floor_paths:
                _fp_prim = _s2.GetPrimAtPath(_fp)
                if _fp_prim.IsValid():
                    UsdShade.MaterialBindingAPI.Apply(_fp_prim).Bind(
                        _floor_mat, UsdShade.Tokens.strongerThanDescendants)
            # Also walk all children of defaultGroundPlane
            _gp = _s2.GetPrimAtPath("/World/defaultGroundPlane")
            if _gp.IsValid():
                from pxr import Usd, UsdGeom
                for _p in Usd.PrimRange(_gp):
                    if _p.IsA(UsdGeom.Gprim) or _p.GetTypeName() == "Mesh":
                        UsdShade.MaterialBindingAPI.Apply(_p).Bind(
                            _floor_mat, UsdShade.Tokens.strongerThanDescendants)
        except Exception as _fe:
            log_message(f"[Floor] Could not set floor color: {_fe}")

        # ── Scene lighting & background: deep navy blue ──────────────────
        try:
            from pxr import UsdLux, Sdf, Gf as _Gf
            import omni.usd as _ousd
            _stage = _ousd.get_context().get_stage()
            # Dome light: deep navy sky
            dome = UsdLux.DomeLight.Define(_stage, "/World/SceneDomeLight")
            dome.CreateIntensityAttr(800.0)
            dome.CreateColorAttr(_Gf.Vec3f(0.04, 0.08, 0.22))   # dark navy
            # Key fill light: soft cool-white from above-front
            key = UsdLux.RectLight.Define(_stage, "/World/KeyLight")
            key.CreateIntensityAttr(6000.0)
            key.CreateColorAttr(_Gf.Vec3f(0.88, 0.92, 1.00))
            key.CreateWidthAttr(3.0)
            key.CreateHeightAttr(2.0)
            xf = key.GetPrim().GetAttribute("xformOp:translate")
            from pxr import UsdGeom as _UG
            xfapi = _UG.XformCommonAPI(key.GetPrim())
            xfapi.SetTranslate(_Gf.Vec3d(0, -3.5, 4.5))
            xfapi.SetRotate(_Gf.Vec3f(50, 0, 0))
            # Background plane (visual sky): large blue quad behind robots
            bg = _UG.Mesh.Define(_stage, "/World/Background")
            bg.GetPointsAttr().Set([
                _Gf.Vec3f(-6, -12, -0.5), _Gf.Vec3f(-6, 12, -0.5),
                _Gf.Vec3f(-6, 12,  5.5),  _Gf.Vec3f(-6, -12,  5.5),
            ])
            bg.GetFaceVertexCountsAttr().Set([4])
            bg.GetFaceVertexIndicesAttr().Set([0, 1, 2, 3])
            from pxr import UsdShade as _US, Sdf as _Sdf
            bg_mat = _US.Material.Define(_stage, "/World/BgMaterial")
            bg_sh  = _US.Shader.Define(_stage, "/World/BgMaterial/Shader")
            bg_sh.CreateIdAttr("UsdPreviewSurface")
            bg_sh.CreateInput("diffuseColor",
                              _Sdf.ValueTypeNames.Color3f).Set(_Gf.Vec3f(0.0, 0.0, 0.0))
            bg_sh.CreateInput("roughness", _Sdf.ValueTypeNames.Float).Set(1.0)
            bg_sh.CreateInput("emissiveColor",
                              _Sdf.ValueTypeNames.Color3f).Set(_Gf.Vec3f(0.0, 0.0, 0.0))
            bg_mat.CreateSurfaceOutput().ConnectToSource(bg_sh.ConnectableAPI(), "surface")
            _US.MaterialBindingAPI.Apply(bg.GetPrim()).Bind(bg_mat)
        except Exception as _e:
            log_message(f"[Scene] Lighting setup note: {_e}")

        # ── Camera: face-to-face, looking straight at the robot front ────
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            # eye is directly in front (negative Y = front of robot),
            # slightly elevated; target is robot center at ~chest height.
            set_camera_view(
                eye=np.array([ 3.5, 0.0, 1.2]),
                target=np.array([0.0, 0.0, 0.8]),
            )
            log_message("[Camera] Set to face-to-face position.")
        except Exception as _ce:
            log_message(f"[Camera] Could not set camera view: {_ce}")

        # If URDF path is provided (no USD), convert URDF -> USD once, then
        # use the resulting USD for all 4 prims.
        self._urdf_mode = False
        if self.robot_urdf_path and not self.robot_usd_path:
            self._urdf_mode = True
            import os, tempfile
            urdf_abs = self.robot_urdf_path
            if not os.path.isabs(urdf_abs):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                root_dir = os.path.abspath(os.path.join(script_dir, "..", "..", "..", ".."))
                urdf_abs = os.path.join(root_dir, self.robot_urdf_path)
            log_message(f"Converting URDF to USD: {urdf_abs}")

            from isaacsim.asset.importer.urdf import _urdf as omni_urdf
            urdf_interface = omni_urdf.acquire_urdf_interface()

            import_config = omni_urdf.ImportConfig()
            import_config.fix_base = True
            import_config.make_default_prim = False
            import_config.merge_fixed_joints = False
            import_config.import_inertia_tensor = True
            import_config.default_drive_type = omni_urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
            import_config.default_drive_strength = 50.0
            import_config.default_position_drive_damping = 1.0
            # Disable USD native instancing so that FBX mesh prims land directly
            # in the stage hierarchy (visible to PrimRange & bindable per-robot).
            # Without this, all 4 robot instances share a single prototype mesh
            # and cannot be given independent material colors.
            try:
                import_config.create_instanceable_usd = False
            except AttributeError:
                pass  # older importer versions may not have this flag

            self._urdf_dir = os.path.dirname(urdf_abs)
            self._urdf_filename = os.path.basename(urdf_abs)
            self._urdf_interface = urdf_interface
            self._urdf_import_config = import_config
            self._urdf_parsed = urdf_interface.parse_urdf(
                self._urdf_dir, self._urdf_filename, import_config
            )
        
        # Create all 4 robots
        self.robots = {}
        self._urdf_inner_root = ""

        for name, prim_path in self.prim_paths.items():
            if self._urdf_mode:
                # Import URDF directly into the live stage.
                # import_robot with a non-file dest_path imports into current stage.
                dest = self._urdf_interface.import_robot(
                    self._urdf_dir,
                    self._urdf_filename,
                    self._urdf_parsed,
                    self._urdf_import_config,
                    prim_path,   # stage destination path
                )
                log_message(f"URDF imported {name} -> dest prim: {dest}")
                # dest is usually the articulation root, e.g. "/genie"
                # The robot ends up at prim_path, with the articulation root
                # as a child.  But import_robot may put it directly at /genie.
                # Check what actually exists:
                from pxr import UsdGeom, Gf
                dest_prim = stage.GetPrimAtPath(dest)
                prim_at_path = stage.GetPrimAtPath(prim_path)

                if dest_prim.IsValid():
                    art_prim_path = dest
                    # If dest is not under prim_path, rename/move
                    if not dest.startswith(prim_path):
                        from isaacsim.core.utils.prims import move_prim
                        # Create the parent Xform
                        if not prim_at_path.IsValid():
                            UsdGeom.Xform.Define(stage, prim_path)
                        target = prim_path + "/" + dest.rsplit("/", 1)[-1]
                        move_prim(dest, target)
                        art_prim_path = target
                        log_message(f"  Moved {dest} -> {target}")
                elif prim_at_path.IsValid():
                    # import_robot put it under prim_path
                    art_prim_path = prim_path
                    # Check if there are children
                    children = list(prim_at_path.GetChildren())
                    if children:
                        art_prim_path = children[0].GetPath().pathString
                        log_message(f"  Using child as art root: {art_prim_path}")
                else:
                    log_message(f"  WARNING: neither {dest} nor {prim_path} is valid!")
                    art_prim_path = prim_path  # fallback

                # Detect the inner root name from the first import
                if not self._urdf_inner_root:
                    self._urdf_inner_root = art_prim_path[len(prim_path):] if art_prim_path.startswith(prim_path) else ""
                    log_message(f"URDF inner root: '{self._urdf_inner_root}'")

                # Apply translation
                translate_prim = stage.GetPrimAtPath(prim_path)
                if not translate_prim.IsValid():
                    # Create parent Xform for positioning
                    UsdGeom.Xform.Define(stage, prim_path)
                    translate_prim = stage.GetPrimAtPath(prim_path)
                if translate_prim.IsValid():
                    xformable = UsdGeom.Xformable(translate_prim)
                    xformable.ClearXformOpOrder()
                    xformable.AddTranslateOp().Set(Gf.Vec3d(*positions[name]))
            else:
                sim_module.create_prim(
                    prim_path=prim_path,
                    prim_type="Xform",
                    usd_path=self.robot_usd_path,
                    translation=positions[name]
                )
                art_prim_path = prim_path
            log_message(f"Created {name}, art_root={art_prim_path}")

            robot = sim_module.Articulation(
                prim_paths_expr=art_prim_path,
                name=f"robot_{name}"
            )
            self.world.scene.add(robot)
            self.robots[name] = robot
        
        # Configure articulation properties
        articulation_props = sim_module.schemas.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            fix_root_link=True,
        )
        for prim_path in self.prim_paths.values():
            art_path = prim_path + self._urdf_inner_root if self._urdf_inner_root else prim_path
            sim_module.schemas.modify_articulation_root_properties(art_path, articulation_props, stage)
        
        # Disable collision between all robots
        self._setup_collision_filtering()

        # Initialize physics FIRST — world.reset() may reload USD prims,
        # so material bindings must be applied AFTER this call.
        self.world.reset()
        for _ in range(5):
            self.world.step(render=False)

        # Save color config so we can reapply after every world.reset()
        # color_cfg: (prim_path, diffuse_rgb, emissive_rgb, opacity, label)
        # Palette tuned for deep blue background:
        #   SIM_noGAP  → bright icy white-cyan  (contrast: luminance)
        #   SIM_withGAP → vivid golden yellow   (contrast: complementary hue)
        #   REAL_1/2   → warm coral-red, semi-transparent ghost
        self._color_cfg = [
            (self.prim_paths['sim_nogap'],
             (1.00, 1.00, 1.00), (0.40, 0.40, 0.40), 1.0,  "SIM_noGAP"),   # pure white
            (self.prim_paths['real_1'],
             (0.00, 0.30, 1.00), (0.00, 0.10, 0.50), 1.0,  "REAL"),         # pure blue solid
            (self.prim_paths['sim_withgap'],
             (1.00, 1.00, 0.00), (0.50, 0.50, 0.00), 1.0,  "SIM_withGAP"), # pure yellow
            (self.prim_paths['real_2'],
             (0.00, 0.30, 1.00), (0.00, 0.10, 0.50), 1.0,  "REAL_2"),      # pure blue solid
        ]
        # Create fallback custom materials
        self._create_robot_materials()
        # Apply colors now: SetInstanceable(False) is called here, which
        # invalidates the PhysX tensor view (physics.tensors simulationView).
        self._apply_robot_colors()

        # world.reset() after de-instancing is MANDATORY: SetInstanceable(False)
        # rewrites USD collision prims, causing PhysX to invalidate the internal
        # _physics_view on all Articulation objects.  A second reset() rebuilds it.
        log_message("Re-initializing physics after de-instancing...")
        self.world.reset()
        for _ in range(3):
            self.world.step(render=False)


        # Get joint indices
        all_dof_names = self.robots['sim_nogap']._dof_names
        log_message(f"Available DOFs: {all_dof_names}")
        
        self.joint_indices = []
        for joint in self.joint_names:
            key = joint.split("/")[-1]
            # Try exact match first
            try:
                index = self.robots['sim_nogap'].get_dof_index(dof_name=key)
                self.joint_indices.append(index)
                continue
            except Exception:
                pass
            # Try suffix match: DOF names from URDF import may have idx## prefix
            matched = False
            for i, dof_name in enumerate(all_dof_names):
                if dof_name.endswith(key) or dof_name.endswith("_" + key):
                    self.joint_indices.append(i)
                    matched = True
                    break
            if not matched:
                log_message(f"Warning: Could not find joint '{key}' in available DOFs")

        # Fallback: if names mismatch but DOF count matches, use first num_dofs joints by index.
        if len(self.joint_indices) == 0 and self.num_dofs <= len(all_dof_names):
            self.joint_indices = list(range(self.num_dofs))
            log_message(
                f"Warning: no joint names matched; fallback to first {self.num_dofs} DOFs by index."
            )
        self.joint_indices = np.array(self.joint_indices)
        log_message(f"Mapped {len(self.joint_indices)} joints: {self.joint_indices}")
        
        # Configure PD controllers for SIM robots
        self._config_controllers()
        
        log_message("Simulation setup complete!")
        log_message("")
        log_message("Layout:")
        log_message("  CENTER (overlapping): [WHITE=SIM_noGAP] [BLUE ghost=REAL] [YELLOW=SIM_withGAP]")
        log_message("  RIGHT (+Y):           [BLUE solid=REAL_2]")
    
    def _setup_collision_filtering(self):
        """Disable collision between all robots."""
        try:
            from pxr import UsdPhysics
            
            # Create collision groups for each robot
            group_paths = {
                'sim_nogap':   "/World/CollisionGroup_SimNoGap",
                'real_1':      "/World/CollisionGroup_Real",
                'sim_withgap': "/World/CollisionGroup_SimWithGap",
                'real_2':      "/World/CollisionGroup_Real2",
            }
            
            groups = {}
            for name, path in group_paths.items():
                groups[name] = UsdPhysics.CollisionGroup.Define(self.stage, path)
            
            # Each group filters out all others
            for name, group in groups.items():
                for other_name, other_path in group_paths.items():
                    if name != other_name:
                        group.GetFilteredGroupsRel().AddTarget(other_path)
            
            # Add robot colliders to their respective groups
            for name, robot_path in self.prim_paths.items():
                robot_prim = self.stage.GetPrimAtPath(robot_path)
                if robot_prim:
                    self._add_to_collision_group(robot_prim, groups[name])
            
            log_message("Collision filtering configured for all 4 robots")
            
        except Exception as e:
            log_message(f"Warning: Could not setup collision filtering: {e}")
    
    def _add_to_collision_group(self, prim, group):
        """Recursively add prim and children to collision group."""
        from pxr import UsdPhysics
        try:
            if prim.HasAPI(UsdPhysics.CollisionAPI):
                group.GetCollidersCollectionAPI().GetIncludesRel().AddTarget(prim.GetPath())
            for child in prim.GetChildren():
                self._add_to_collision_group(child, group)
        except:
            pass
    
    # ──────────────────────────────────────────────────────────────────────────
    # Robot colorization
    # Core strategy: instead of fighting USD material binding composition rules,
    # we directly mutate the shader inputs on whichever material is ALREADY
    # bound to each Gprim.  The URDF importer creates OmniPBR materials and
    # binds them; we just change their diffuse_color_constant in-place.
    # displayColor is also set as a non-RTX fallback.
    # ──────────────────────────────────────────────────────────────────────────

    def _create_robot_materials(self):
        """Create one UsdPreviewSurface material per robot.

        Mirrors sim_dual_runner.py: simple UsdPreviewSurface, one per robot.
        """
        from pxr import UsdShade, Sdf, Gf
        self._robot_mat_paths = {}
        self._isaac_materials = {}

        for prim_path, color, emissive, opacity, label in self._color_cfg:
            safe = prim_path.strip("/").replace("/", "_")
            mat_path = f"/World/RobotColors/{safe}"

            # Remove stale prim if a previous run left one
            existing = self.stage.GetPrimAtPath(mat_path)
            if existing.IsValid():
                self.stage.RemovePrim(mat_path)

            mat = UsdShade.Material.Define(self.stage, mat_path)
            shader = UsdShade.Shader.Define(self.stage, f"{mat_path}/Shader")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor",
                               Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness",     Sdf.ValueTypeNames.Float).Set(0.6)
            shader.CreateInput("metallic",      Sdf.ValueTypeNames.Float).Set(0.0)
            shader.CreateInput("emissiveColor",
                               Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*emissive))
            if opacity < 1.0:
                shader.CreateInput("opacity",
                                   Sdf.ValueTypeNames.Float).Set(opacity)
            mat.CreateSurfaceOutput().ConnectToSource(
                shader.ConnectableAPI(), "surface")

            self._robot_mat_paths[prim_path] = mat_path
            log_message(f"[Color] {label}: UsdPreviewSurface at {mat_path}")

    def _create_robot_materials_usd(self):
        """Alias kept for call-site compat — delegates to _create_robot_materials."""
        self._create_robot_materials()

    def _apply_robot_colors(self):
        """Apply per-robot colors. Mirrors sim_dual_runner.py exactly:

        1. SetInstanceable(False) on every instanceable Xform under each robot
           → makes USD expand shared FBX prototypes into unique per-robot copies
           → enables independent material binding per robot
        2. UnbindAllBindings() + Bind(UsdPreviewSurface, strongerThanDescendants)
           on every Mesh/Gprim prim (now accessible after de-instancing)
        3. Fallback: same link-path approach as sim_dual_runner._set_robot_ghost()
        """
        from pxr import Usd, UsdGeom, UsdShade, Gf, Sdf

        first_call = not getattr(self, "_colors_applied_once", False)
        self._colors_applied_once = True

        if not getattr(self, "_robot_mat_paths", None):
            self._create_robot_materials()

        # Link names with visual FBX geometry (from agibot_g1.urdf)
        all_link_names = [
            "base_link", "body_link1", "body_link2",
            "head_link1", "head_link2",
            "arm_l_base_link", "arm_r_base_link",
            "arm_l_link1", "arm_l_link2", "arm_l_link3",
            "arm_l_link4", "arm_l_link5", "arm_l_link6", "arm_l_end_link",
            "arm_r_link1", "arm_r_link2", "arm_r_link3",
            "arm_r_link4", "arm_r_link5", "arm_r_link6", "arm_r_end_link",
            "gripper_l_base_link",
            "gripper_l_inner_link1", "gripper_l_inner_link2",
            "gripper_l_inner_link3", "gripper_l_inner_link4",
            "gripper_l_outer_link1", "gripper_l_outer_link2",
            "gripper_l_outer_link3", "gripper_l_outer_link4",
            "gripper_r_base_link",
            "gripper_r_inner_link1", "gripper_r_inner_link2",
            "gripper_r_inner_link3", "gripper_r_inner_link4",
            "gripper_r_outer_link1", "gripper_r_outer_link2",
            "gripper_r_outer_link3", "gripper_r_outer_link4",
        ]

        for prim_path, color, emissive, opacity, label in self._color_cfg:
            mat_path = (self._robot_mat_paths or {}).get(prim_path)
            if not mat_path:
                continue
            mat_prim = self.stage.GetPrimAtPath(mat_path)
            if not mat_prim.IsValid():
                continue
            robot_mat = UsdShade.Material(mat_prim)

            root = self.stage.GetPrimAtPath(prim_path)
            if not root.IsValid():
                if first_call:
                    log_message(f"[Color] {label}: root invalid at {prim_path}")
                continue

            # ── Step 1: De-instance FBX Xforms (only on first call) ──────
            # SetInstanceable(False) causes USD to treat this robot's FBX
            # Xforms as unique — no longer sharing a prototype with other
            # robots. Exact same effect as UrdfFileCfg(make_instanceable=False)
            # used in sim_dual_runner.py.
            # Two-pass: collect first, then modify (safe during PrimRange iter).
            de_count = 0
            if first_call:
                instanceable_prims = [
                    p for p in Usd.PrimRange(root) if p.IsInstanceable()
                ]
                for p in instanceable_prims:
                    p.SetInstanceable(False)
                    de_count += 1
                log_message(f"[Color] {label}: de-instanced {de_count} Xforms")

            session_layer = self.stage.GetSessionLayer()
            gf_color   = Gf.Vec3f(*color)
            gf_emissive_top = Gf.Vec3f(*emissive)

            # ── Step 2: Directly overwrite EVERY Shader prim's color ─────
            # This is the most reliable approach: instead of fighting binding
            # priority, we directly mutate the color inputs on all Shader
            # prims found under this robot in the session layer.
            # Session layer overrides beat ALL sublayer opinions including
            # FBX-embedded materials — no competition, no flicker.
            shader_count = 0
            with Usd.EditContext(self.stage, session_layer):
                for prim in Usd.PrimRange(root):
                    if not prim.IsA(UsdShade.Shader):
                        continue
                    shader = UsdShade.Shader(prim)
                    gf_emissive = gf_emissive_top
                    # OmniPBR MDL inputs
                    for iname, itype, ival in [
                        ("diffuse_color_constant",
                         Sdf.ValueTypeNames.Color3f, gf_color),
                        ("reflection_roughness_constant",
                         Sdf.ValueTypeNames.Float,   0.6),
                        ("metallic_constant",
                         Sdf.ValueTypeNames.Float,   0.0),
                        ("enable_emission",
                         Sdf.ValueTypeNames.Bool,    True),
                        ("emissive_color",
                         Sdf.ValueTypeNames.Color3f, gf_emissive),
                        ("emissive_intensity",
                         Sdf.ValueTypeNames.Float,   3000.0),
                    ]:
                        try:
                            shader.CreateInput(iname, itype).Set(ival)
                        except Exception:
                            pass
                    # UsdPreviewSurface inputs
                    for iname, itype, ival in [
                        ("diffuseColor",
                         Sdf.ValueTypeNames.Color3f, gf_color),
                        ("emissiveColor",
                         Sdf.ValueTypeNames.Color3f, gf_emissive),
                        ("roughness",
                         Sdf.ValueTypeNames.Float,   0.6),
                        ("metallic",
                         Sdf.ValueTypeNames.Float,   0.0),
                    ]:
                        try:
                            shader.CreateInput(iname, itype).Set(ival)
                        except Exception:
                            pass
                    shader_count += 1
            if first_call:
                log_message(
                    f"[Color] {label}: mutated {shader_count} shader prims")

            # ── Step 3: Bind our flat material (belt + suspenders) ────────
            # This covers any prims that had no pre-existing material bound.
            bound_count = 0
            with Usd.EditContext(self.stage, session_layer):
                try:
                    UsdShade.MaterialBindingAPI.Apply(root).Bind(
                        robot_mat, UsdShade.Tokens.strongerThanDescendants)
                except Exception:
                    pass
                for prim in Usd.PrimRange(root):
                    if prim.GetTypeName() == "Mesh" or prim.IsA(UsdGeom.Gprim):
                        try:
                            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                                robot_mat,
                                UsdShade.Tokens.strongerThanDescendants)
                            bound_count += 1
                        except Exception:
                            pass
            if first_call:
                log_message(
                    f"[Color] {label}: session-layer bound {bound_count} mesh prims")

            # ── displayColor for non-RTX / preview modes ──────────────────
            for prim in Usd.PrimRange(root):
                if prim.IsA(UsdGeom.Gprim):
                    try:
                        UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([gf_color])
                        UsdGeom.Gprim(prim).GetDisplayOpacityAttr().Set([opacity])
                    except Exception:
                        pass



    def _recolor_material(self, mat, gf_color, opacity, label):
        """(Deprecated — kept for API compat. Colors are now managed via
        _create_robot_materials / _apply_robot_colors.)"""
        pass

    def _bind_custom_material(self, prim_path, color, opacity, label, root):
        """(Deprecated — kept for API compat. Colors are now managed via
        _create_robot_materials / _apply_robot_colors.)"""
        pass

    # Keep thin wrappers so call sites in _setup_simulation still compile
    def _set_robot_color(self, prim_path, color, opacity, label=""):
        pass  # handled by _create_robot_materials + _apply_robot_colors

    def _set_color_recursive(self, prim, color, opacity):
        from pxr import UsdGeom, Gf
        if prim.IsA(UsdGeom.Gprim):
            UsdGeom.Gprim(prim).GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        for child in prim.GetChildren():
            self._set_color_recursive(child, color, opacity)


    
    def _config_controllers(self):
        """Configure PD controllers for SIM robots."""
        kps = np.array(self.joint_kp, dtype=np.float32)
        kds = np.array(self.joint_kd, dtype=np.float32)
        
        for name in ['sim_nogap', 'sim_withgap']:
            robot = self.robots[name]
            robot.set_effort_modes("force", joint_indices=self.joint_indices)
            robot.switch_control_mode("position", joint_indices=self.joint_indices)
            robot.set_gains(kps=kps, kds=kds, joint_indices=self.joint_indices)
        
        log_message(f"Configured PD controllers: Kp={kps}, Kd={kds}")
    
    def _init_video_writer(self):
        """Initialize video writer using viewport capture."""
        import omni.ui
        video_path = self.video_path
        if not video_path:
            os.makedirs("outputs/videos", exist_ok=True)
            video_path = f"outputs/videos/gaponet_comparison_{self.robot_name}_motion{self.motion_idx}.mp4"
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)

        viewport = omni.ui.Workspace.get_window("Viewport")
        self.viewport_api = viewport.viewport_api
        frame_width = self.viewport_api.resolution[0]
        frame_height = self.viewport_api.resolution[1]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.video_writer = cv2.VideoWriter(video_path, fourcc, float(self.video_fps), (frame_width, frame_height))
        log_message(f"Video writer initialized: {video_path} ({frame_width}x{frame_height} @ {self.video_fps} fps)")

    def _capture_video_frame(self, *args, **kwargs):
        """Callback to capture a single video frame from the viewport."""
        capsule, data_size, width, height = args[0], args[1], args[2], args[3]
        ptr = ctypes.pythonapi.PyCapsule_GetPointer(capsule, None)
        buffer = ctypes.string_at(ptr, data_size)
        raw_array = np.frombuffer(buffer, dtype=np.uint8)
        image_array = raw_array.reshape((height, width, 4))
        rgb_image = cv2.cvtColor(image_array[:, :, :3], cv2.COLOR_RGB2BGR)
        self.video_writer.write(rgb_image)

    def run_visualization(self, max_frames=-1):
        """Run the visualization loop."""
        if max_frames > 0:
            self.num_frames = min(self.num_frames, max_frames)
        log_message("Starting GAPONet comparison visualization...")
        log_message(f"Running {self.num_frames} frames...")
        
        # Reset
        self.world.reset()
        self._config_controllers()
        # Do warm-up render steps FIRST so that FBX visual assets are fully
        # loaded into the USD stage (FBX references are resolved lazily on
        # first render).  THEN apply colors so PrimRange actually finds meshes.
        for _ in range(30):
            self.world.step(render=True)
        # Force re-run diagnostic so we can see how many meshes were found
        # after the warm-up render steps.
        self._colors_applied_once = False
        self._apply_robot_colors()

        # Buffer phase
        BUFFER_TIME = 2.0
        buffer_steps = int(BUFFER_TIME / self.control_dt)
        
        # Get initial positions
        initial_pos = self.robots['sim_nogap'].get_joint_positions(joint_indices=self.joint_indices)[0]
        target_pos = self.commands[0]
        
        log_message("Buffer phase: moving to initial position...")
        
        # Pre-allocate arrays
        num_total_dof = self.robots['sim_nogap'].num_dof
        target_nogap = np.zeros((1, num_total_dof), dtype=np.float32)
        target_withgap = np.zeros((1, num_total_dof), dtype=np.float32)
        real_array = np.zeros((1, num_total_dof), dtype=np.float32)
        
        # Buffer phase
        for step in range(buffer_steps * self.divisor):
            if step % self.divisor == 0:
                alpha = (step // self.divisor) / buffer_steps
                interp_pos = (1 - alpha) * initial_pos + alpha * target_pos
                
                # Set SIM robots targets
                target_nogap.fill(0)
                target_withgap.fill(0)
                for j, idx in enumerate(self.joint_indices):
                    target_nogap[0, idx] = interp_pos[j]
                    target_withgap[0, idx] = interp_pos[j] + self.delta_actions[0, j]
                
                self.robots['sim_nogap'].set_joint_position_targets(target_nogap)
                self.robots['sim_withgap'].set_joint_position_targets(target_withgap)
            
            # Set REAL robots positions
            real_array.fill(0)
            for j, idx in enumerate(self.joint_indices):
                real_array[0, idx] = self.real_positions[0, j]
            self.robots['real_1'].set_joint_positions(real_array)
            self.robots['real_2'].set_joint_positions(real_array)

            
            self.world.step(render=True)
        
        # Initialize video recording if requested
        if self.save_video:
            from omni.kit.viewport.utility import capture_viewport_to_buffer
            self._init_video_writer()

        log_message("Main execution: showing comparison...")
        
        # Track errors
        errors_nogap = []
        errors_withgap = []
        
        # Main execution loop
        for counter in range(self.num_frames * self.divisor):
            frame_idx = counter // self.divisor
            
            if frame_idx >= self.num_frames:
                break
            
            if counter % self.divisor == 0:
                command = self.commands[frame_idx]
                delta = self.delta_actions[frame_idx]
                real_pos = self.real_positions[frame_idx]
                
                # SIM without GAPONet: use command directly
                target_nogap.fill(0)
                for j, idx in enumerate(self.joint_indices):
                    target_nogap[0, idx] = command[j]
                self.robots['sim_nogap'].set_joint_position_targets(target_nogap)
                
                # SIM with GAPONet: use command + delta
                target_withgap.fill(0)
                for j, idx in enumerate(self.joint_indices):
                    target_withgap[0, idx] = command[j] + delta[j]
                self.robots['sim_withgap'].set_joint_position_targets(target_withgap)
                
                # Update real positions
                real_array.fill(0)
                for j, idx in enumerate(self.joint_indices):
                    real_array[0, idx] = real_pos[j]
                
                # Log progress
                if frame_idx % 100 == 0:
                    sim_nogap_pos = self.robots['sim_nogap'].get_joint_positions(joint_indices=self.joint_indices)[0]
                    sim_withgap_pos = self.robots['sim_withgap'].get_joint_positions(joint_indices=self.joint_indices)[0]
                    
                    error_nogap = np.abs(sim_nogap_pos - real_pos).mean()
                    error_withgap = np.abs(sim_withgap_pos - real_pos).mean()
                    
                    errors_nogap.append(error_nogap)
                    errors_withgap.append(error_withgap)
                    
                    log_message(
                        f"Frame {frame_idx}/{self.num_frames} | "
                        f"Error w/o GAPONet: {np.rad2deg(error_nogap):.2f}° | "
                        f"Error w/ GAPONet: {np.rad2deg(error_withgap):.2f}°"
                    )
            
            # ALWAYS set REAL robot positions before step
            self.robots['real_1'].set_joint_positions(real_array)
            self.robots['real_2'].set_joint_positions(real_array)

            self.world.step(render=True)

            # Capture video frame
            if self.save_video and self.video_writer is not None:
                capture_viewport_to_buffer(self.viewport_api, self._capture_video_frame)
        
        # Print summary
        if errors_nogap:
            log_message("")
            log_message("=" * 60)
            log_message("SUMMARY")
            log_message("=" * 60)
            log_message(f"Mean error WITHOUT GAPONet: {np.rad2deg(np.mean(errors_nogap)):.2f}°")
            log_message(f"Mean error WITH GAPONet: {np.rad2deg(np.mean(errors_withgap)):.2f}°")
            if np.mean(errors_nogap) > 0:
                improvement = (np.mean(errors_nogap) - np.mean(errors_withgap)) / np.mean(errors_nogap) * 100
                log_message(f"Improvement: {improvement:.1f}%")
        
        # Release video writer
        if self.save_video and self.video_writer is not None:
            self.video_writer.release()
            log_message(f"Video saved to: {self.video_path or 'outputs/videos/gaponet_comparison_*.mp4'}")

        log_message("")
        log_message("Visualization complete!")
        
        if not self.args.headless:
            log_message("Press Ctrl+C to exit...")
            try:
                while True:
                    self.world.step(render=True)
            except KeyboardInterrupt:
                pass


def main():
    parser = argparse.ArgumentParser(description="Visualize GAPONet comparison with 4 robots")
    parser.add_argument("--robot-name", type=str, default="so101", help="Robot name")
    parser.add_argument("--precomputed-file", type=str, required=True, 
                        help="Path to precomputed delta actions file (.npz) from precompute_delta_actions.py")
    parser.add_argument("--group-offset", type=float, default=1.8, help="Extra gap between the two groups (m)")
    parser.add_argument("--physics-freq", type=int, default=200, help="Physics frequency")
    parser.add_argument("--render-freq", type=int, default=200, help="Render frequency")
    parser.add_argument("--control-freq", type=int, default=50, help="Control frequency")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--livestream", type=int, default=2, help="Livestream type")
    parser.add_argument("--max-frames", type=int, default=-1, help="Max frames to run (-1=all)")
    parser.add_argument("--save-video", action="store_true", default=False, help="Save video of the visualization")
    parser.add_argument("--video-path", type=str, default=None, help="Output video file path (default: outputs/videos/gaponet_comparison_<robot>_motion<idx>.mp4)")
    parser.add_argument("--video-fps", type=int, default=30, help="Video FPS (default: 30)")

    args = parser.parse_args()

    # Initialize SimulationApp
    from isaacsim.simulation_app import SimulationApp
    app = SimulationApp({"headless": args.headless, "livestream": args.livestream})

    # Import Isaac Sim modules
    import carb
    import isaaclab.sim.schemas as schemas
    from isaacsim.core.api import World
    from isaacsim.core.prims import Articulation
    from isaacsim.core.utils.prims import create_prim
    from isaacsim.core.utils.stage import add_reference_to_stage, create_new_stage

    # Create simulation module container
    class SimModule:
        pass
    
    sim_module = SimModule()
    sim_module.schemas = schemas
    sim_module.World = World
    sim_module.Articulation = Articulation
    sim_module.add_reference_to_stage = add_reference_to_stage
    sim_module.create_new_stage = create_new_stage
    sim_module.create_prim = create_prim

    # Setup assets module
    import assets as assets_module
    assets_module.carb = carb

    # Run visualization
    visualizer = GAPONetComparisonVisualizer(args, sim_module)
    visualizer.run_visualization(max_frames=args.max_frames)

    app.close()


if __name__ == "__main__":
    main()

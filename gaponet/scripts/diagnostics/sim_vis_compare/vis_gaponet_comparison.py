"""
GAPONet 效果对比可视化脚本

显示 4 个机器人并排：
- Group 1 (左): SIM without GAPONet (蓝色) + REAL (橙色半透明)
- Group 2 (右): SIM with GAPONet (绿色) + REAL (橙色半透明)

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
import os
from datetime import datetime

import numpy as np


def log_message(message):
    """Format and print log messages with timestamp."""
    current_time = datetime.now().strftime("%H:%M:%S")
    print(f"[VisGAPO][Log][{current_time}] {message}")


class GAPONetComparisonVisualizer:
    """
    Visualize GAPONet comparison with 4 robots:
    - SIM without GAPONet + REAL (left group)  
    - SIM with GAPONet + REAL (right group)
    
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
        
        # Get PD gains
        self.joint_kp = self.robot_config.get_config_value("joint_kp", None)
        self.joint_kd = self.robot_config.get_config_value("joint_kd", None)
        if self.joint_kp is None:
            self.joint_kp = [100.0] * 6
        if self.joint_kd is None:
            self.joint_kd = [10.0] * 6
        log_message(f"Using PD gains: Kp={self.joint_kp}, Kd={self.joint_kd}")
        
        # Default joint names for SO101
        self.joint_names = ["Rotation", "Pitch", "Elbow", "Wrist_Pitch", "Wrist_Roll", "Jaw"]
        
        # Load precomputed data
        self._load_precomputed_data()
        
        # Setup simulation with 4 robots
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
        
        log_message(f"Motion index: {self.motion_idx}")
        log_message(f"Loaded {self.num_frames} frames with {self.num_dofs} DOFs")
        log_message(f"Delta action range: [{self.delta_actions.min():.4f}, {self.delta_actions.max():.4f}] rad")
        log_message(f"Mean |delta|: {np.abs(self.delta_actions).mean():.4f} rad ({np.rad2deg(np.abs(self.delta_actions).mean()):.2f}°)")
    
    def _setup_simulation(self):
        """Set up simulation with 4 robots."""
        import omni
        
        sim_module = self.sim_module
        
        # Robot paths
        self.prim_paths = {
            'sim_nogap': "/World/Robot_SIM_noGAP",
            'real_1': "/World/Robot_REAL_1",
            'sim_withgap': "/World/Robot_SIM_withGAP",
            'real_2': "/World/Robot_REAL_2",
        }
        
        self.robot_usd_path = self.robot_config.usd_path
        base_offset = list(self.robot_config.offset)
        
        # Robot positions:
        # Group 1 (left, without GAPONet): x = -group_offset/2
        # Group 2 (right, with GAPONet): x = +group_offset/2
        positions = {
            'sim_nogap': [base_offset[0] - self.group_offset/2, base_offset[1], base_offset[2]],
            'real_1': [base_offset[0] - self.group_offset/2, base_offset[1], base_offset[2]],
            'sim_withgap': [base_offset[0] + self.group_offset/2, base_offset[1], base_offset[2]],
            'real_2': [base_offset[0] + self.group_offset/2, base_offset[1], base_offset[2]],
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
        
        # Create all 4 robots
        self.robots = {}
        for name, prim_path in self.prim_paths.items():
            sim_module.create_prim(
                prim_path=prim_path,
                prim_type="Xform",
                usd_path=self.robot_usd_path,
                translation=positions[name]
            )
            log_message(f"Created {name} at {prim_path}")
            
            robot = sim_module.Articulation(
                prim_paths_expr=prim_path,
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
            sim_module.schemas.modify_articulation_root_properties(prim_path, articulation_props, stage)
        
        # Disable collision between all robots
        self._setup_collision_filtering()
        
        # Set robot colors
        self._set_robot_color(self.prim_paths['sim_nogap'], color=(0.2, 0.4, 1.0), opacity=1.0)   # Blue
        self._set_robot_color(self.prim_paths['real_1'], color=(1.0, 0.4, 0.1), opacity=0.6)      # Orange
        self._set_robot_color(self.prim_paths['sim_withgap'], color=(0.2, 0.8, 0.2), opacity=1.0) # Green
        self._set_robot_color(self.prim_paths['real_2'], color=(1.0, 0.4, 0.1), opacity=0.6)      # Orange
        
        # Initialize physics
        self.world.reset()
        for _ in range(5):
            self.world.step(render=False)
        
        # Get joint indices
        all_dof_names = self.robots['sim_nogap']._dof_names
        log_message(f"Available DOFs: {all_dof_names}")
        
        self.joint_indices = []
        for joint in self.joint_names:
            key = joint.split("/")[-1]
            try:
                index = self.robots['sim_nogap'].get_dof_index(dof_name=key)
                self.joint_indices.append(index)
            except Exception as e:
                log_message(f"Warning: Could not find joint '{key}': {e}")
        self.joint_indices = np.array(self.joint_indices)
        log_message(f"Mapped {len(self.joint_indices)} joints: {self.joint_indices}")
        
        # Configure PD controllers for SIM robots
        self._config_controllers()
        
        log_message("Simulation setup complete!")
        log_message("")
        log_message("Layout:")
        log_message("  LEFT GROUP (without GAPONet):  BLUE=SIM, ORANGE=REAL")
        log_message("  RIGHT GROUP (with GAPONet):    GREEN=SIM, ORANGE=REAL")
    
    def _setup_collision_filtering(self):
        """Disable collision between all robots."""
        try:
            from pxr import UsdPhysics
            
            # Create collision groups for each robot
            group_paths = {
                'sim_nogap': "/World/CollisionGroup_SimNoGap",
                'real_1': "/World/CollisionGroup_Real1",
                'sim_withgap': "/World/CollisionGroup_SimWithGap",
                'real_2': "/World/CollisionGroup_Real2",
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
    
    def _set_robot_color(self, prim_path, color, opacity):
        """Set robot color and opacity."""
        try:
            from pxr import UsdGeom, Gf
            
            prim = self.stage.GetPrimAtPath(prim_path)
            if not prim:
                return
            
            self._set_color_recursive(prim, color, opacity)
            log_message(f"Set color for {prim_path}: RGB={color}, opacity={opacity}")
        except Exception as e:
            log_message(f"Could not set color for {prim_path}: {e}")
    
    def _set_color_recursive(self, prim, color, opacity):
        """Recursively set color on all mesh descendants."""
        from pxr import UsdGeom, Gf
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            mesh.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
            mesh.GetDisplayOpacityAttr().Set([opacity])
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
    
    def run_visualization(self):
        """Run the visualization loop."""
        log_message("Starting GAPONet comparison visualization...")
        log_message(f"Running {self.num_frames} frames...")
        
        # Reset
        self.world.reset()
        self._config_controllers()
        
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
        
        log_message("")
        log_message("Visualization complete! Press Ctrl+C to exit...")
        
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
    parser.add_argument("--group-offset", type=float, default=0.5, help="Offset between the two groups")
    parser.add_argument("--physics-freq", type=int, default=200, help="Physics frequency")
    parser.add_argument("--render-freq", type=int, default=200, help="Render frequency")
    parser.add_argument("--control-freq", type=int, default=50, help="Control frequency")
    parser.add_argument("--headless", action="store_true", help="Run headless")
    parser.add_argument("--livestream", type=int, default=2, help="Livestream type")

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
    visualizer.run_visualization()

    app.close()


if __name__ == "__main__":
    main()

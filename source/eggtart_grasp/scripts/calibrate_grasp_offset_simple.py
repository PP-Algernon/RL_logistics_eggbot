#!/usr/bin/env python3
"""
Compute grasp offset from jaw geometry and display visual markers.

This is a simplified version that:
1. Loads the robot in simulation
2. Computes jaw centroids from mesh geometry
3. Shows visual markers for current grasp point, jaw centroids, and EE frame
4. Prints the current offset and suggested auto-computed offset
5. Runs for a few seconds then exits

For interactive tuning, run without --headless to use keyboard controls.

Usage:
    ./isaaclab.sh -p source/eggtart_grasp/scripts/calibrate_grasp_offset_simple.py
    ./isaaclab.sh -p source/eggtart_grasp/scripts/calibrate_grasp_offset_simple.py --headless  # auto mode
"""

import argparse
import numpy as np
import torch
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Grasp point calibration tool")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, SPHERE_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply
from scipy.spatial.transform import Rotation as R

import eggtart_grasp.tasks.mobile_grasp
from eggtart_grasp.assets import EGGTART_CFG, EGGTART_EE_GRASP_OFFSET, EGGTART_NOMINAL_JOINT_POS


def load_stl_centroid(stl_path: Path) -> np.ndarray:
    """Load binary STL and compute centroid."""
    import struct
    with open(stl_path, 'rb') as f:
        f.read(80)  # header
        num_tri = struct.unpack('<I', f.read(4))[0]
        verts = []
        for _ in range(num_tri):
            f.read(12)  # normal
            for __ in range(3):
                verts.append(struct.unpack('<fff', f.read(12)))
            f.read(2)  # attr
    return np.array(verts).mean(axis=0)


def compute_jaw_offsets():
    """Compute jaw centroids in their respective body frames."""
    script_dir = Path(__file__).resolve().parent
    urdf_dir = script_dir.parent / "eggtart_grasp" / "assets" / "urdf"

    # Mesh centroids in mesh-local frames (mm -> m)
    fixed_c_mesh = load_stl_centroid(urdf_dir / "meshes" / "part_034_solid_034.stl") * 0.001
    moving_c_mesh = load_stl_centroid(urdf_dir / "meshes" / "part_035_solid_035.stl") * 0.001

    # URDF visual origins and rotations
    fixed_origin = np.array([0.33609379, 0.47285302, -0.21011169])
    fixed_rpy = np.array([1.57079633, 0.0, 0.0])
    moving_origin = np.array([0.31597274, 0.46294983, -0.18567585])
    moving_rpy = np.array([1.57079633, 0.0, 0.0])

    # Transform to body frames
    fixed_rot = R.from_euler('xyz', fixed_rpy).as_matrix()
    moving_rot = R.from_euler('xyz', moving_rpy).as_matrix()

    fixed_offset_link005 = fixed_origin + fixed_rot @ fixed_c_mesh
    moving_offset_ee = moving_origin + moving_rot @ moving_c_mesh

    return fixed_offset_link005, moving_offset_ee


def main():
    # Compute jaw centroids
    fixed_jaw_link005, moving_jaw_ee = compute_jaw_offsets()

    print("\n" + "="*70)
    print("GRASP POINT CALIBRATION")
    print("="*70)
    print(f"\nFixed jaw centroid (link_005 frame):    {fixed_jaw_link005}")
    print(f"Moving jaw centroid (end_effector frame): {moving_jaw_ee}")
    print(f"\nCurrent EGGTART_EE_GRASP_OFFSET: {EGGTART_EE_GRASP_OFFSET}")

    # Create simulation
    sim_cfg = sim_utils.SimulationCfg(dt=1/120, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.5, 1.5, 1.2], target=[0.0, 0.0, 0.3])

    # Create scene
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)
    scene_cfg.terrain = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    scene_cfg.robot = EGGTART_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.robot.init_state.pos = (0.0, 0.0, 0.0)

    scene = InteractiveScene(scene_cfg)
    sim.reset()

    # Set nominal pose
    robot: Articulation = scene["robot"]
    joint_pos = robot.data.default_joint_pos.clone()
    for name_pattern, pos in EGGTART_NOMINAL_JOINT_POS.items():
        joint_indices, _ = robot.find_joints(name_pattern)
        if len(joint_indices) > 0:
            joint_pos[0, joint_indices] = pos

    robot.write_joint_state_to_sim(joint_pos, robot.data.default_joint_vel)
    scene.reset()

    # Setup markers - use independent paths, we'll position them manually in EE frame
    ee_cfg = SceneEntityCfg("robot", body_names=["end_effector"])
    link005_cfg = SceneEntityCfg("robot", body_names=["link_005"])

    # Current grasp point (red sphere)
    grasp_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/grasp_point")
    grasp_cfg.markers["sphere"].radius = 0.012
    grasp_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
    grasp_marker = VisualizationMarkers(grasp_cfg)

    # EE frame
    frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/ee_frame")
    frame_cfg.markers["frame"].scale = (0.08, 0.08, 0.08)
    ee_frame_marker = VisualizationMarkers(frame_cfg)

    # Fixed jaw (blue)
    fixed_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/fixed_jaw")
    fixed_cfg.markers["sphere"].radius = 0.008
    fixed_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
    fixed_jaw_marker = VisualizationMarkers(fixed_cfg)

    # Moving jaw (cyan)
    moving_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/moving_jaw")
    moving_cfg.markers["sphere"].radius = 0.008
    moving_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 1.0)
    moving_jaw_marker = VisualizationMarkers(moving_cfg)

    # Auto-computed midpoint (green)
    auto_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/auto_grasp")
    auto_cfg.markers["sphere"].radius = 0.010
    auto_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
    auto_marker = VisualizationMarkers(auto_cfg)

    # Convert numpy to torch
    fixed_jaw_torch = torch.tensor(fixed_jaw_link005, device=robot.device, dtype=torch.float32)
    moving_jaw_torch = torch.tensor(moving_jaw_ee, device=robot.device, dtype=torch.float32)
    current_offset = torch.tensor(EGGTART_EE_GRASP_OFFSET, device=robot.device, dtype=torch.float32)

    # Get body indices
    ee_body_idx = ee_cfg.body_ids[0] if isinstance(ee_cfg.body_ids, list) else 0
    link005_body_idx = link005_cfg.body_ids[0] if isinstance(link005_cfg.body_ids, list) else 0

    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=robot.device)

    print("\n" + "="*70)
    print("VISUAL MARKERS (look in the viewport):")
    print("  RED sphere    = Current grasp point (from EGGTART_EE_GRASP_OFFSET)")
    print("  BLUE sphere   = Fixed jaw centroid (on link_005)")
    print("  CYAN sphere   = Moving jaw centroid (on end_effector)")
    print("  GREEN sphere  = Auto-computed grasp point (midpoint of jaws)")
    print("  RGB frame     = End-effector coordinate frame")
    print("="*70)

    # Simulation loop
    sim_dt = sim.get_physics_dt()
    count = 0
    max_steps = 500 if args_cli.headless else 3000

    print(f"\nRunning for {max_steps} steps...")

    while simulation_app.is_running() and count < max_steps:
        # Step simulation first
        scene.sim.step()
        scene.update(sim_dt)

        # Update markers every frame - compute world positions from EE local offsets
        # Get poses
        ee_pos_w = robot.data.body_pos_w[0, ee_body_idx]
        ee_quat_w = robot.data.body_quat_w[0, ee_body_idx]
        link005_pos_w = robot.data.body_pos_w[0, link005_body_idx]
        link005_quat_w = robot.data.body_quat_w[0, link005_body_idx]

        # Current grasp point: EE local offset -> world position
        current_grasp_w = ee_pos_w + quat_apply(ee_quat_w, current_offset)
        grasp_marker.visualize(current_grasp_w.unsqueeze(0), identity_quat)

        # EE frame at EE origin
        ee_frame_marker.visualize(ee_pos_w.unsqueeze(0), ee_quat_w.unsqueeze(0))

        # Moving jaw: EE local offset -> world position
        moving_jaw_w = ee_pos_w + quat_apply(ee_quat_w, moving_jaw_torch)
        moving_jaw_marker.visualize(moving_jaw_w.unsqueeze(0), identity_quat)

        # Fixed jaw: link_005 local offset -> world position
        fixed_jaw_w = link005_pos_w + quat_apply(link005_quat_w, fixed_jaw_torch)
        fixed_jaw_marker.visualize(fixed_jaw_w.unsqueeze(0), identity_quat)

        # Auto-computed grasp: midpoint in world, then report in EE local
        auto_grasp_w = (fixed_jaw_w + moving_jaw_w) / 2.0
        auto_marker.visualize(auto_grasp_w.unsqueeze(0), identity_quat)

        # Print auto offset once after stabilization (in EE local frame)
        if count == 120:
            # Transform world midpoint to EE local frame
            ee_quat_conj = torch.tensor(
                [ee_quat_w[0], -ee_quat_w[1], -ee_quat_w[2], -ee_quat_w[3]],
                device=robot.device
            )
            auto_grasp_ee_local = quat_apply(ee_quat_conj, auto_grasp_w - ee_pos_w)

            print(f"\n{'='*70}")
            print("AUTO-COMPUTED OFFSET (from jaw geometry):")
            print(f"  EGGTART_EE_GRASP_OFFSET = {tuple(auto_grasp_ee_local.cpu().numpy())}")
            print(f"\nDistance from current offset: {torch.norm(auto_grasp_ee_local - current_offset).item():.4f} m")
            print(f"{'='*70}\n")

        count += 1

    print("\nDone. Check the visual markers in the viewport.")
    print("For interactive tuning, run the full calibrate_grasp_offset.py without --headless\n")

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print(f"Error: {e}")
        traceback.print_exc()
        simulation_app.close()

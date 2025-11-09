"""
Script to visualize trajectories from HDF5 datasets.

Usage:
    # View a single trajectory
    python scripts/visualize_trajectory.py --input data/example-data.hdf5 --trajectory 0
    
    # Compare original vs blurred
    python scripts/visualize_trajectory.py \
        --input data/example-data.hdf5 \
        --input-blurred data/example-data-blurred.hdf5 \
        --trajectory 0
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


def play_trajectory(hdf5_path, trajectory_idx=0, fps=30, window_name="Trajectory"):
    """
    Play a trajectory from an HDF5 file.
    
    Args:
        hdf5_path: Path to HDF5 file
        trajectory_idx: Index of trajectory to play (default: 0)
        fps: Frames per second for playback (default: 30)
        window_name: Name of the display window
    """
    with h5py.File(hdf5_path, 'r') as f:
        trajectory_keys = list(f.keys())
        
        if trajectory_idx >= len(trajectory_keys):
            print(f"Error: Trajectory index {trajectory_idx} out of range. Available: 0-{len(trajectory_keys)-1}")
            return
        
        traj_key = trajectory_keys[trajectory_idx]
        print(f"Playing trajectory: {traj_key}")
        
        obs = f[traj_key]['obs'][:]  # Shape: (T, H, W, C)
        num_frames = obs.shape[0]
        
        print(f"Trajectory shape: {obs.shape}")
        print(f"Number of frames: {num_frames}")
        print(f"Press 'q' to quit, 'p' to pause/resume, 'r' to restart")
        
        frame_idx = 0
        paused = False
        delay = int(1000 / fps)  # milliseconds per frame
        
        while True:
            if not paused:
                frame = obs[frame_idx]
                
                # Convert BGR to RGB if needed (OpenCV uses BGR)
                if frame.shape[-1] == 3:
                    frame_display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_display = frame
                
                # Add frame counter
                frame_display = frame_display.copy()
                # cv2.putText(
                #     frame_display, 
                #     f"Frame: {frame_idx}/{num_frames-1}", 
                #     (10, 30), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 
                #     0.7, 
                #     (255, 255, 255), 
                #     2
                # )
                
                cv2.imshow(window_name, frame_display)
                
                frame_idx += 1
                if frame_idx >= num_frames:
                    frame_idx = 0  # Loop
            
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):
                frame_idx = 0
                print("Restarted")
        
        cv2.destroyWindow(window_name)


def compare_trajectories(original_path, blurred_path, trajectory_idx=0, fps=30):
    """
    Play original and blurred trajectories side-by-side.
    
    Args:
        original_path: Path to original HDF5 file
        blurred_path: Path to blurred HDF5 file
        trajectory_idx: Index of trajectory to play (default: 0)
        fps: Frames per second for playback (default: 30)
    """
    with h5py.File(original_path, 'r') as f_orig, h5py.File(blurred_path, 'r') as f_blur:
        trajectory_keys_orig = list(f_orig.keys())
        trajectory_keys_blur = list(f_blur.keys())
        
        if trajectory_idx >= len(trajectory_keys_orig):
            print(f"Error: Trajectory index {trajectory_idx} out of range in original. Available: 0-{len(trajectory_keys_orig)-1}")
            return
        
        if trajectory_idx >= len(trajectory_keys_blur):
            print(f"Error: Trajectory index {trajectory_idx} out of range in blurred. Available: 0-{len(trajectory_keys_blur)-1}")
            return
        
        traj_key_orig = trajectory_keys_orig[trajectory_idx]
        traj_key_blur = trajectory_keys_blur[trajectory_idx]
        
        print(f"Comparing trajectories:")
        print(f"  Original: {traj_key_orig}")
        print(f"  Blurred:  {traj_key_blur}")
        
        obs_orig = f_orig[traj_key_orig]['obs'][:]
        obs_blur = f_blur[traj_key_blur]['obs'][:]
        
        num_frames = min(obs_orig.shape[0], obs_blur.shape[0])
        
        print(f"Original shape: {obs_orig.shape}")
        print(f"Blurred shape:  {obs_blur.shape}")
        print(f"Number of frames: {num_frames}")
        print(f"Press 'q' to quit, 'p' to pause/resume, 'r' to restart")
        
        frame_idx = 0
        paused = False
        delay = int(1000 / fps)
        
        while True:
            if not paused:
                frame_orig = obs_orig[frame_idx]
                frame_blur = obs_blur[frame_idx]
                
                # Convert BGR to RGB if needed
                if frame_orig.shape[-1] == 3:
                    frame_orig_display = cv2.cvtColor(frame_orig, cv2.COLOR_RGB2BGR)
                    frame_blur_display = cv2.cvtColor(frame_blur, cv2.COLOR_RGB2BGR)
                else:
                    frame_orig_display = frame_orig
                    frame_blur_display = frame_blur
                
                # Add labels
                frame_orig_display = frame_orig_display.copy()
                frame_blur_display = frame_blur_display.copy()
                
                # cv2.putText(
                #     frame_orig_display, 
                #     f"Original - Frame: {frame_idx}/{num_frames-1}", 
                #     (10, 30), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 
                #     0.7, 
                #     (255, 255, 255), 
                #     2
                # )
                
                # cv2.putText(
                #     frame_blur_display, 
                #     f"Blurred - Frame: {frame_idx}/{num_frames-1}", 
                #     (10, 30), 
                #     cv2.FONT_HERSHEY_SIMPLEX, 
                #     0.7, 
                #     (255, 255, 255), 
                #     2
                # )
                
                # Concatenate horizontally
                combined = np.hstack([frame_orig_display, frame_blur_display])
                
                cv2.imshow("Original vs Blurred", combined)
                
                frame_idx += 1
                if frame_idx >= num_frames:
                    frame_idx = 0  # Loop
            
            key = cv2.waitKey(delay) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):
                frame_idx = 0
                print("Restarted")
        
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize trajectories from HDF5 datasets"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input HDF5 file'
    )
    parser.add_argument(
        '--input-blurred',
        type=str,
        default=None,
        help='Path to blurred HDF5 file for comparison (optional)'
    )
    parser.add_argument(
        '--trajectory',
        type=int,
        default=0,
        help='Index of trajectory to visualize (default: 0)'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Frames per second for playback (default: 30)'
    )
    parser.add_argument(
        '--list',
        action='store_true',
        help='List all trajectories in the dataset and exit'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # List trajectories if requested
    if args.list:
        with h5py.File(args.input, 'r') as f:
            trajectory_keys = list(f.keys())
            print(f"Found {len(trajectory_keys)} trajectories in {args.input}:")
            for idx, key in enumerate(trajectory_keys):
                obs_shape = f[key]['obs'].shape
                print(f"  [{idx}] {key}: {obs_shape[0]} frames, shape {obs_shape}")
        return
    
    # Play trajectory or compare
    if args.input_blurred is None:
        # Single trajectory playback
        play_trajectory(args.input, args.trajectory, args.fps)
    else:
        # Compare original vs blurred
        if not Path(args.input_blurred).exists():
            print(f"Error: Blurred file not found: {args.input_blurred}")
            return
        
        compare_trajectories(args.input, args.input_blurred, args.trajectory, args.fps)


if __name__ == '__main__':
    main()
"""
Script to blur all observations in an HDF5 dataset using depth-based background blurring.

Usage:
    python scripts/blur_hdf5_data.py --input data/example-data.hdf5 --output data/example-data-blurred.hdf5
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm

# Add Depth-Anything-V2 to path
sys.path.insert(0, 'Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2


class DepthBlurProcessor:
    """Applies depth-based background blurring using DepthAnything V2."""
    
    def __init__(self, encoder='vitb', percentile=50, blur_kernel=(11, 11), blur_sigma=5,
                 mask_kernel=(5, 5), mask_sigma=1, threshold_alpha=0.1, device='cuda'):
        """
        Args:
            encoder: DepthAnything encoder size ('vits', 'vitb', 'vitl', 'vitg')
            percentile: Depth percentile threshold for foreground/background separation
            blur_kernel: Kernel size for Gaussian blur on background
            blur_sigma: Sigma for Gaussian blur on background
            mask_kernel: Kernel size for mask feathering
            mask_sigma: Sigma for mask feathering
            threshold_alpha: EMA smoothing factor for depth threshold (0=no update, 1=full update)
            device: Device to run depth model on
        """
        self.percentile = percentile
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.mask_kernel = mask_kernel
        self.mask_sigma = mask_sigma
        self.threshold_alpha = threshold_alpha
        self.device = device
        
        # Model configurations
        model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        # Load DepthAnything model
        self.model = DepthAnythingV2(**model_configs[encoder])
        checkpoint_path = f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'
        # Load checkpoint to the target device directly to avoid device mismatch
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        self.model = self.model.to(device).eval()
        
        print(f"Loaded DepthAnything V2 ({encoder}) for depth-based blurring")
    
    def blur_single_frame(self, frame, depth_threshold=None):
        """
        Apply depth-based blur to a single RGB frame.
        
        Args:
            frame: numpy array of shape (H, W, 3) in range [0, 255] uint8
            depth_threshold: Optional previous threshold value for EMA smoothing.
        
        Returns:
            tuple: (blurred_frame, depth_threshold) where:
                - blurred_frame: numpy array of same shape as input in range [0, 255] uint8
                - depth_threshold: the updated threshold value (smoothed via EMA)
        """
        # Infer depth for this frame
        with torch.no_grad():
            depth = self.model.infer_image(frame)  # HxW depth map
        
        # Normalize depth to [0, 255]
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        
        # Compute current frame's threshold
        current_threshold = np.percentile(depth, self.percentile)
        
        # Apply exponential moving average for smooth threshold adaptation
        # EMA formula: new_value = alpha * current + (1 - alpha) * previous
        if depth_threshold is None:
            # First frame: use current threshold
            depth_threshold = current_threshold
        else:
            # Subsequent frames: smooth with EMA
            depth_threshold = (self.threshold_alpha * current_threshold +
                             (1 - self.threshold_alpha) * depth_threshold)
        
        # Create mask: foreground (high depth values) vs background (low depth values)
        # Note: In DepthAnything, higher depth values = closer to camera = foreground
        mask = (depth > depth_threshold).astype(np.float32)
        
        # Apply mask smoothing for gradual foreground/background transition
        # This creates a softer blend so background objects remain partially visible
        # mask = cv2.GaussianBlur(mask, self.mask_kernel, self.mask_sigma)
        mask3 = mask[..., None]  # Add channel dimension
        
        # Apply softer Gaussian blur to background
        # Reduced blur strength allows model to perceive background motion/objects
        blurred = cv2.GaussianBlur(frame, self.blur_kernel, self.blur_sigma)
        
        # Composite: smooth blend between sharp foreground and softly blurred background
        result = (frame * mask3 + blurred * (1 - mask3)).astype(np.uint8)
        
        return result, depth_threshold
    
    def blur_observation(self, obs, depth_threshold=None):
        """
        Apply depth-based blur to an observation (single frame, RGB).
        
        Args:
            obs: numpy array of shape (H, W, 3) in range [0, 255] uint8
            depth_threshold: Optional fixed threshold value
        
        Returns:
            tuple: (blurred_obs, depth_threshold)
        """
        return self.blur_single_frame(obs, depth_threshold)


def blur_hdf5_dataset(input_path, output_path, encoder='vitb', percentile=30,
                      blur_kernel=(3, 3), blur_sigma=3, mask_kernel=(21, 21),
                      mask_sigma=10, threshold_alpha=0.1, device='cuda', batch_size=32):
    """
    Read an HDF5 dataset, blur all observations, and save to a new HDF5 file.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file
        encoder: DepthAnything encoder size
        percentile: Depth percentile threshold
        blur_kernel: Kernel size for Gaussian blur
        blur_sigma: Sigma for Gaussian blur
        mask_kernel: Kernel size for mask feathering
        mask_sigma: Sigma for mask feathering
        threshold_alpha: EMA smoothing factor for depth threshold
        device: Device to run depth model on
        batch_size: Number of frames to process at once (for progress tracking)
    """
    print(f"Reading dataset from: {input_path}")
    print(f"Output will be saved to: {output_path}")
    
    # Initialize blur processor
    blur_processor = DepthBlurProcessor(
        encoder=encoder,
        percentile=percentile,
        blur_kernel=blur_kernel,
        blur_sigma=blur_sigma,
        mask_kernel=mask_kernel,
        mask_sigma=mask_sigma,
        threshold_alpha=threshold_alpha,
        device=device
    )
    
    # Open input file and create output file
    with h5py.File(input_path, 'r') as input_file:
        with h5py.File(output_path, 'w') as output_file:
            # Copy attributes
            for attr_name, attr_value in input_file.attrs.items():
                output_file.attrs[attr_name] = attr_value
            
            # Process each trajectory
            trajectory_keys = list(input_file.keys())
            print(f"Found {len(trajectory_keys)} trajectories")
            
            for traj_key in tqdm(trajectory_keys, desc="Processing trajectories"):
                traj_group = input_file[traj_key]
                output_traj_group = output_file.create_group(traj_key)
                
                # Get observations
                obs = traj_group['obs'][:]  # Shape: (T, H, W, C)
                num_frames = obs.shape[0]
                
                print(f"\nTrajectory {traj_key}: {num_frames} frames, shape {obs.shape}")
                
                # Blur each observation with moving average threshold
                blurred_obs = []
                depth_threshold = None
                thresholds = []
                for i in tqdm(range(num_frames), desc=f"Blurring {traj_key}", leave=False):
                    frame = obs[i]  # (H, W, C)
                    blurred_frame, depth_threshold = blur_processor.blur_observation(frame, depth_threshold)
                    blurred_obs.append(blurred_frame)
                    thresholds.append(depth_threshold)
                
                print(f"  Depth threshold range: [{min(thresholds):.2f}, {max(thresholds):.2f}], "
                      f"mean: {np.mean(thresholds):.2f}, final: {depth_threshold:.2f}")
                
                # Stack and save blurred observations
                blurred_obs = np.stack(blurred_obs, axis=0)
                output_traj_group.create_dataset('obs', data=blurred_obs, compression='gzip')
                
                # Copy other datasets (actions, states, etc.) without modification
                for dataset_name in traj_group.keys():
                    if dataset_name != 'obs':
                        output_traj_group.create_dataset(
                            dataset_name, 
                            data=traj_group[dataset_name][:],
                            compression='gzip'
                        )
    
    print(f"\nBlurred dataset saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Blur all observations in an HDF5 dataset using depth-based background blurring"
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to input HDF5 file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Path to output HDF5 file'
    )
    parser.add_argument(
        '--encoder', 
        type=str, 
        default='vitb',
        choices=['vits', 'vitb', 'vitl', 'vitg'],
        help='DepthAnything encoder size (default: vitb)'
    )
    parser.add_argument(
        '--percentile', 
        type=int, 
        default=30,
        help='Depth percentile threshold for foreground/background separation (default: 50)'
    )
    parser.add_argument(
        '--blur-kernel',
        type=int,
        nargs=2,
        default=[7, 7],
        help='Kernel size for Gaussian blur on background (default: 7 7). '
             'Smaller values = gentler blur, background more visible.'
    )
    parser.add_argument(
        '--blur-sigma',
        type=float,
        default=2.0,
        help='Sigma for Gaussian blur on background (default: 2.0). '
             'Smaller values = gentler blur, background more visible.'
    )
    parser.add_argument(
        '--mask-kernel',
        type=int,
        nargs=2,
        default=[5, 5],
        help='Kernel size for mask feathering (default: 5 5). '
             'Smaller values = sharper foreground/background boundary.'
    )
    parser.add_argument(
        '--mask-sigma',
        type=float,
        default=1.0,
        help='Sigma for mask feathering (default: 1.0). '
             'Smaller values = sharper foreground/background boundary.'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'mps', 'cpu'],
        help='Device to run depth model on (default: cuda). Use "mps" for Apple Silicon GPUs.'
    )
    parser.add_argument(
        '--threshold-alpha',
        type=float,
        default=0.1,
        help='EMA smoothing factor for depth threshold (default: 0.1). '
             'Lower values = smoother transitions, higher values = faster adaptation.'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for processing (default: 32)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if output file already exists
    if Path(args.output).exists():
        response = input(f"Output file {args.output} already exists. Overwrite? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
    
    # Process the dataset
    blur_hdf5_dataset(
        input_path=args.input,
        output_path=args.output,
        encoder=args.encoder,
        percentile=args.percentile,
        blur_kernel=tuple(args.blur_kernel),
        blur_sigma=args.blur_sigma,
        mask_kernel=tuple(args.mask_kernel),
        mask_sigma=args.mask_sigma,
        threshold_alpha=args.threshold_alpha,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
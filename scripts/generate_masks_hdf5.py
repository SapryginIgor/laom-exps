"""
Script to generate depth-based foreground masks for all observations in an HDF5 dataset.

Usage:
    python scripts/generate_masks_hdf5.py --input data/example-data.hdf5 --output data/example-data-masked.hdf5
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


class DepthMaskGenerator:
    """Generates foreground masks using DepthAnything V2."""
    
    def __init__(self, encoder='vitb', percentile=50, mask_kernel=(21, 21), mask_sigma=10, 
                 threshold_alpha=0.1, device='cuda'):
        """
        Args:
            encoder: DepthAnything encoder size ('vits', 'vitb', 'vitl', 'vitg')
            percentile: Depth percentile threshold for foreground/background separation
            mask_kernel: Kernel size for mask feathering
            mask_sigma: Sigma for mask feathering
            threshold_alpha: EMA smoothing factor for depth threshold (0=no update, 1=full update)
            device: Device to run depth model on
        """
        self.percentile = percentile
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
        
        # Check if checkpoint exists
        import os
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Checkpoint not found at {checkpoint_path}. "
                f"Please download it from: https://huggingface.co/depth-anything/Depth-Anything-V2-{encoder.upper()}/resolve/main/depth_anything_v2_{encoder}.pth"
            )
        
        # Load checkpoint to the target device directly to avoid device mismatch
        try:
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load checkpoint from {checkpoint_path}. "
                f"The file may be corrupted. Please re-download it.\nError: {str(e)}"
            )
        
        self.model = self.model.to(device).eval()
        print(f"Loaded DepthAnything V2 ({encoder}) for mask generation")
    
    def generate_mask_single_frame(self, frame, depth_threshold=None):
        """
        Generate foreground mask for a single RGB frame.
        
        Args:
            frame: numpy array of shape (H, W, 3) in range [0, 255] uint8
            depth_threshold: Optional previous threshold value for EMA smoothing.
        
        Returns:
            tuple: (mask, depth_threshold) where:
                - mask: numpy array of shape (H, W, 3) in range [0, 1] float32
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
        if depth_threshold is None:
            depth_threshold = current_threshold
        else:
            depth_threshold = (self.threshold_alpha * current_threshold +
                             (1 - self.threshold_alpha) * depth_threshold)
        
        # Create mask: foreground (high depth values) = 1, background (low depth values) = 0
        mask = (depth > depth_threshold).astype(np.float32)
        
        # Feather mask edges for smoother transitions
        mask = cv2.GaussianBlur(mask, self.mask_kernel, self.mask_sigma)
        
        # Keep as single channel (H, W) - we'll expand to 3 channels during loading
        # This saves 3x storage space
        return mask, depth_threshold
    
    def generate_mask_observation(self, obs, depth_threshold=None):
        """
        Generate mask for an observation (single frame, RGB).
        
        Args:
            obs: numpy array of shape (H, W, 3) in range [0, 255] uint8
            depth_threshold: Optional fixed threshold value
        
        Returns:
            tuple: (mask, depth_threshold)
        """
        return self.generate_mask_single_frame(obs, depth_threshold)


def generate_masks_hdf5_dataset(input_path, output_path, encoder='vitb', percentile=50,
                                mask_kernel=(21, 21), mask_sigma=10, threshold_alpha=0.1,
                                device='cuda', batch_size=32):
    """
    Read an HDF5 dataset, generate masks for all observations, and save to a separate HDF5 file.
    This keeps the original data file unchanged and stores masks separately to avoid large file sizes.
    
    Args:
        input_path: Path to input HDF5 file
        output_path: Path to output HDF5 file (will contain only masks)
        encoder: DepthAnything encoder size
        percentile: Depth percentile threshold
        mask_kernel: Kernel size for mask feathering
        mask_sigma: Sigma for mask feathering
        threshold_alpha: EMA smoothing factor for depth threshold
        device: Device to run depth model on
        batch_size: Number of frames to process at once (for progress tracking)
    """
    print(f"Reading dataset from: {input_path}")
    print(f"Masks will be saved to: {output_path}")
    
    # Initialize mask generator
    mask_generator = DepthMaskGenerator(
        encoder=encoder,
        percentile=percentile,
        mask_kernel=mask_kernel,
        mask_sigma=mask_sigma,
        threshold_alpha=threshold_alpha,
        device=device
    )
    
    # Open input file and create output file (masks only)
    with h5py.File(input_path, 'r') as input_file:
        with h5py.File(output_path, 'w') as output_file:
            # Copy only essential attributes
            output_file.attrs['img_hw'] = input_file.attrs['img_hw']
            output_file.attrs['source_file'] = input_path
            output_file.attrs['encoder'] = encoder
            output_file.attrs['percentile'] = percentile
            
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
                
                # Generate mask for each observation with moving average threshold
                masks = []
                depth_threshold = None
                thresholds = []
                for i in tqdm(range(num_frames), desc=f"Generating masks for {traj_key}", leave=False):
                    frame = obs[i]  # (H, W, C)
                    mask, depth_threshold = mask_generator.generate_mask_observation(frame, depth_threshold)
                    masks.append(mask)
                    thresholds.append(depth_threshold)
                
                print(f"  Depth threshold range: [{min(thresholds):.2f}, {max(thresholds):.2f}], "
                      f"mean: {np.mean(thresholds):.2f}, final: {depth_threshold:.2f}")
                
                # Stack and save masks
                masks = np.stack(masks, axis=0)  # (T, H, W) - single channel
                print(f"  Mask coverage: mean={masks.mean():.3f}, min={masks.min():.3f}, max={masks.max():.3f}")
                
                # Convert to uint8 [0, 255] to save space (4x smaller than float32)
                # We'll convert back to float [0, 1] during loading
                masks_uint8 = (masks * 255).astype(np.uint8)
                
                # Save masks as uint8 with compression
                output_traj_group.create_dataset('masks', data=masks_uint8, compression='gzip', dtype='uint8')
                

    
    print(f"\nMasks saved to: {output_path}")
    print(f"Original data remains in: {input_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth-based foreground masks for all observations in an HDF5 dataset"
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
        default=50,
        help='Depth percentile threshold for foreground/background separation (default: 50)'
    )
    parser.add_argument(
        '--mask-kernel',
        type=int,
        nargs=2,
        default=[21, 21],
        help='Kernel size for mask feathering (default: 21 21). '
             'Creates smooth foreground/background transition.'
    )
    parser.add_argument(
        '--mask-sigma',
        type=float,
        default=10.0,
        help='Sigma for mask feathering (default: 10.0). '
             'Creates smooth foreground/background transition.'
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
    generate_masks_hdf5_dataset(
        input_path=args.input,
        output_path=args.output,
        encoder=args.encoder,
        percentile=args.percentile,
        mask_kernel=tuple(args.mask_kernel),
        mask_sigma=args.mask_sigma,
        threshold_alpha=args.threshold_alpha,
        device=args.device,
        batch_size=args.batch_size
    )


if __name__ == '__main__':
    main()
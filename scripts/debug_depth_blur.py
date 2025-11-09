"""
Debug script to visualize depth maps and blurring masks.

Usage:
    python scripts/debug_depth_blur.py --input data/example-data.hdf5 --trajectory 0 --frame 0
"""

import argparse
import sys
from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch

# Add Depth-Anything-V2 to path
sys.path.insert(0, 'Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2


def visualize_depth_and_mask(hdf5_path, trajectory_idx=0, frame_idx=0, 
                             encoder='vitb', percentile=50, device='cuda'):
    """
    Visualize the depth map, mask, and blurring result for debugging.
    """
    # Load model
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    model = DepthAnythingV2(**model_configs[encoder])
    checkpoint_path = f'Depth-Anything-V2/checkpoints/depth_anything_v2_{encoder}.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device).eval()
    
    # Load frame
    with h5py.File(hdf5_path, 'r') as f:
        trajectory_keys = list(f.keys())
        traj_key = trajectory_keys[trajectory_idx]
        obs = f[traj_key]['obs'][:]
        frame = obs[frame_idx]
    
    print(f"Frame shape: {frame.shape}")
    print(f"Frame dtype: {frame.dtype}")
    print(f"Frame range: [{frame.min()}, {frame.max()}]")
    
    # Infer depth
    with torch.no_grad():
        depth = model.infer_image(frame)
    
    print(f"\nRaw depth shape: {depth.shape}")
    print(f"Raw depth dtype: {depth.dtype}")
    print(f"Raw depth range: [{depth.min():.4f}, {depth.max():.4f}]")
    print(f"Raw depth mean: {depth.mean():.4f}")
    print(f"Raw depth std: {depth.std():.4f}")
    
    # Normalize depth to [0, 255]
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth_normalized = depth_normalized.astype(np.uint8)
    
    # Create mask with current logic
    p = np.percentile(depth_normalized, percentile)
    mask_current = (depth_normalized > p).astype(np.float32)
    
    # Create mask with inverted logic
    mask_inverted = (depth_normalized < p).astype(np.float32)
    
    print(f"\nPercentile {percentile}: {p}")
    print(f"Current mask (depth > p): {mask_current.mean():.2%} foreground")
    print(f"Inverted mask (depth < p): {mask_inverted.mean():.2%} foreground")
    
    # Apply blur with both masks
    blur_kernel = (11, 11)
    blur_sigma = 5
    mask_kernel = (3, 3)
    mask_sigma = 1
    
    blurred = cv2.GaussianBlur(frame, blur_kernel, blur_sigma)
    
    # Current logic
    mask_current_smooth = cv2.GaussianBlur(mask_current, mask_kernel, mask_sigma)
    mask_current_smooth3 = mask_current_smooth[..., None]
    result_current = (frame * mask_current_smooth3 + blurred * (1 - mask_current_smooth3)).astype(np.uint8)
    
    # Inverted logic
    mask_inverted_smooth = cv2.GaussianBlur(mask_inverted, mask_kernel, mask_sigma)
    mask_inverted_smooth3 = mask_inverted_smooth[..., None]
    result_inverted = (frame * mask_inverted_smooth3 + blurred * (1 - mask_inverted_smooth3)).astype(np.uint8)
    
    # Visualize
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    # Row 1: Original, Depth, Blurred
    axes[0, 0].imshow(frame)
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(depth_normalized, cmap='viridis')
    axes[0, 1].set_title(f'Depth Map (normalized)\nBrighter = Higher Depth Value')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(blurred)
    axes[0, 2].set_title('Fully Blurred')
    axes[0, 2].axis('off')
    
    # Row 2: Current logic
    axes[1, 0].imshow(mask_current, cmap='gray')
    axes[1, 0].set_title(f'Current Mask (depth > {p:.0f})\nWhite = Foreground (kept sharp)')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(mask_current_smooth, cmap='gray')
    axes[1, 1].set_title('Current Mask (smoothed)')
    axes[1, 1].axis('off')
    
    axes[1, 2].imshow(result_current)
    axes[1, 2].set_title('Result with Current Logic')
    axes[1, 2].axis('off')
    
    # Row 3: Inverted logic
    axes[2, 0].imshow(mask_inverted, cmap='gray')
    axes[2, 0].set_title(f'Inverted Mask (depth < {p:.0f})\nWhite = Foreground (kept sharp)')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(mask_inverted_smooth, cmap='gray')
    axes[2, 1].set_title('Inverted Mask (smoothed)')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(result_inverted)
    axes[2, 2].set_title('Result with Inverted Logic')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig('depth_debug.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to: depth_debug.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Debug depth-based blurring by visualizing depth maps and masks"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input HDF5 file'
    )
    parser.add_argument(
        '--trajectory',
        type=int,
        default=0,
        help='Index of trajectory (default: 0)'
    )
    parser.add_argument(
        '--frame',
        type=int,
        default=0,
        help='Index of frame within trajectory (default: 0)'
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
        help='Depth percentile threshold (default: 50)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'mps', 'cpu'],
        help='Device to run depth model on (default: cuda)'
    )
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        return
    
    visualize_depth_and_mask(
        args.input,
        args.trajectory,
        args.frame,
        args.encoder,
        args.percentile,
        args.device
    )


if __name__ == '__main__':
    main()
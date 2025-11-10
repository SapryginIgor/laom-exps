import os
import random

import gymnasium as gym
import h5py
import numpy as np
import torch
import torch.nn as nn
from shimmy import DmControlCompatibilityV0
from torch.utils.data import Dataset, IterableDataset

from .dcs import suite


def set_seed(seed, env=None, deterministic_torch=False):
    if env is not None:
        env.seed(seed)
        env.action_space.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(deterministic_torch)


def get_optim_groups(model, weight_decay):
    return [
        # do not decay biases and single-column parameters (rmsnorm), those are usually scales
        {"params": (p for p in model.parameters() if p.dim() < 2), "weight_decay": 0.0},
        {"params": (p for p in model.parameters() if p.dim() >= 2), "weight_decay": weight_decay},
    ]


def get_grad_norm(model):
    grads = [param.grad.detach().flatten() for param in model.parameters() if param.grad is not None]
    norm = torch.cat(grads).norm()
    return norm


def soft_update(target, source, tau=1e-3):
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)


class DCSInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu"):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]

        self.frame_stack = frame_stack
        self.traj_len = self.observations[0].shape[0]

    def __get_padded_obs(self, traj_idx, idx):
        # stacking frames
        # : is not inclusive, so +1 is needed
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        # pad if at the beginning as in the wrapper (with the first frame)
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)

        return obs

    def __len__(self):
        return len(self.actions) * (self.traj_len - 1)

    def __getitem__(self, idx):
        traj_idx, transition_idx = divmod(idx, self.traj_len - 1)

        obs = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
        action = self.actions[traj_idx][transition_idx]

        return obs, next_obs, action


class DCSLAOMInMemoryDataset(Dataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1, load_masks=False, masks_path=None):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.states = [torch.tensor(df[traj]["states"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]
            self.state_dim = self.states[0][0].shape[-1]
            
            # Load pre-generated masks from separate file if specified
            self.masks = None
            if load_masks:
                # If masks_path not provided, try to infer it from data path
                if masks_path is None:
                    # Try common naming patterns: data.hdf5 -> data-masks.hdf5
                    import os
                    base_path = os.path.splitext(hdf5_path)[0]
                    masks_path = f"{base_path}-masks.hdf5"
                
                try:
                    with h5py.File(masks_path, "r") as masks_file:
                        # Load masks as uint8 and convert to float32 [0, 1]
                        # Masks are stored as single-channel (T, H, W) uint8 [0, 255]
                        self.masks = [
                            torch.tensor(masks_file[traj]["masks"][:], device=device).float() / 255.0
                            for traj in df.keys()
                        ]
                        print(f"Loaded pre-generated masks from {masks_path}")
                except (FileNotFoundError, KeyError) as e:
                    print(f"Warning: Could not load masks from {masks_path}. Error: {e}")
                    print(f"Masks will not be used. Generate them with: python scripts/generate_masks_hdf5.py --input {hdf5_path} --output {masks_path}")
                    load_masks = False

        self.frame_stack = frame_stack
        self.traj_len = self.observations[0].shape[0]
        assert 1 <= max_offset < self.traj_len
        self.max_offset = max_offset
        self.load_masks = load_masks

    def __get_padded_obs(self, traj_idx, idx):
        # stacking frames
        # : is not inclusive, so +1 is needed
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        # pad if at the beginning as in the wrapper (with the first frame)
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)

        return obs
    
    def __get_padded_mask(self, traj_idx, idx):
        """Get padded mask for frame stacking (same logic as obs)."""
        if self.masks is None:
            return None
        
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1
        mask = self.masks[traj_idx][min_obs_idx:max_obs_idx]  # (frame_stack, H, W)
        
        # pad if at the beginning
        if mask.shape[0] < self.frame_stack:
            pad_mask = mask[0][None]
            mask = torch.concat([pad_mask for _ in range(self.frame_stack - mask.shape[0])] + [mask])
        
        # Expand single-channel mask to 3 channels and reshape to match obs format
        # mask is (frame_stack, H, W), we need (H, W, frame_stack*3)
        mask = mask.unsqueeze(-1)  # (frame_stack, H, W, 1)
        mask = mask.expand(-1, -1, -1, 3)  # (frame_stack, H, W, 3)
        mask = mask.permute((1, 2, 0, 3))  # (H, W, frame_stack, 3)
        mask = mask.reshape(*mask.shape[:2], -1)  # (H, W, frame_stack*3)
        
        return mask

    def __len__(self):
        return len(self.actions) * (self.traj_len - self.max_offset)

    def __getitem__(self, idx):
        traj_idx, transition_idx = divmod(idx, self.traj_len - self.max_offset)
        action = self.actions[traj_idx][transition_idx]
        state = self.states[traj_idx][transition_idx]

        obs = self.__get_padded_obs(traj_idx, transition_idx)
        next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
        offset = random.randint(1, self.max_offset)
        future_obs = self.__get_padded_obs(traj_idx, transition_idx + offset)
        
        # Only return masks if explicitly requested (for backward compatibility)
        if self.load_masks:
            next_obs_mask = self.__get_padded_mask(traj_idx, transition_idx + 1)
            return obs, next_obs, future_obs, action, state, (offset - 1), next_obs_mask
        else:
            # Original behavior: return 6 elements without mask
            return obs, next_obs, future_obs, action, state, (offset - 1)


class DCSLAOMTrueActionsDataset(IterableDataset):
    def __init__(self, hdf5_path, frame_stack=1, device="cpu", max_offset=1):
        with h5py.File(hdf5_path, "r") as df:
            self.observations = [torch.tensor(df[traj]["obs"][:], device=device) for traj in df.keys()]
            self.actions = [torch.tensor(df[traj]["actions"][:], device=device) for traj in df.keys()]
            self.states = [torch.tensor(df[traj]["states"][:], device=device) for traj in df.keys()]
            self.img_hw = df.attrs["img_hw"]
            self.act_dim = self.actions[0][0].shape[-1]
            self.state_dim = self.states[0][0].shape[-1]

        self.frame_stack = frame_stack
        self.traj_len = self.observations[0].shape[0]
        assert 1 <= max_offset < self.traj_len
        self.max_offset = max_offset

    def __get_padded_obs(self, traj_idx, idx):
        # stacking frames
        # : is not inclusive, so +1 is needed
        min_obs_idx = max(0, idx - self.frame_stack + 1)
        max_obs_idx = idx + 1
        obs = self.observations[traj_idx][min_obs_idx:max_obs_idx]

        # pad if at the beginning as in the wrapper (with the first frame)
        if obs.shape[0] < self.frame_stack:
            pad_img = obs[0][None]
            obs = torch.concat([pad_img for _ in range(self.frame_stack - obs.shape[0])] + [obs])
        # TODO: check this one more time...
        obs = obs.permute((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)
        return obs

    def __iter__(self):
        while True:
            traj_idx = random.randint(0, len(self.actions) - 1)
            transition_idx = random.randint(0, self.actions[traj_idx].shape[0] - self.max_offset)

            obs = self.__get_padded_obs(traj_idx, transition_idx)
            next_obs = self.__get_padded_obs(traj_idx, transition_idx + 1)
            offset = random.randint(1, self.max_offset)
            future_obs = self.__get_padded_obs(traj_idx, transition_idx + offset)

            action = self.actions[traj_idx][transition_idx]
            state = self.states[traj_idx][transition_idx]

            yield obs, next_obs, future_obs, action, state, (offset - 1)


def normalize_img(img):
    return ((img / 255.0) - 0.5) * 2.0


def unnormalize_img(img):
    return ((img / 2.0) + 0.5) * 255.0


def weight_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.orthogonal_(m.weight.data)
        if hasattr(m.bias, "data"):
            m.bias.data.fill_(0.0)


class SelectPixelsObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = self.env.observation_space["pixels"]

    def observation(self, obs):
        return obs["pixels"]


class FlattenStackedFrames(gym.ObservationWrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_shape = self.env.observation_space.shape
        new_shape = old_shape[1:-1] + (old_shape[0] * old_shape[-1],)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=new_shape, dtype=np.uint8)

    def observation(self, obs):
        obs = obs.transpose((1, 2, 0, 3))
        obs = obs.reshape(*obs.shape[:2], -1)
        return obs


class DepthBlurWrapper(gym.ObservationWrapper):
    """Applies depth-based background blurring to observations during evaluation."""
    
    def __init__(self, env: gym.Env, encoder='vits', percentile=50,
                 blur_kernel=(7, 7), blur_sigma=2.0, device='cuda'):
        super().__init__(env)
        import sys
        import cv2
        import torch
        
        # Add Depth-Anything-V2 to path if not already there
        if 'Depth-Anything-V2' not in sys.path[0]:
            sys.path.insert(0, 'Depth-Anything-V2')
        from depth_anything_v2.dpt import DepthAnythingV2
        
        self.percentile = percentile
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.device = device
        self.depth_threshold = None  # EMA threshold across frames
        
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
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
        self.model = self.model.to(device).eval()
        
        print(f"DepthBlurWrapper: Loaded DepthAnything V2 ({encoder}) for evaluation-time blurring")
    
    def observation(self, obs):
        """Apply depth-based blur to observation."""
        import cv2
        import torch
        import numpy as np
        
        # obs shape depends on whether frame stacking is applied
        # Could be (H, W, C) or (H, W, C*frames) after FlattenStackedFrames
        original_shape = obs.shape
        channels = obs.shape[-1]
        
        # Process each frame if stacked (assuming 3 channels per frame)
        frames = channels // 3
        blurred_frames = []
        
        for f in range(frames):
            # Extract frame
            frame = obs[:, :, f*3:(f+1)*3]
            
            # Infer depth
            with torch.no_grad():
                depth = self.model.infer_image(frame)
            
            # Normalize depth
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            # Compute threshold
            current_threshold = np.percentile(depth, self.percentile)
            if self.depth_threshold is None:
                self.depth_threshold = current_threshold
            else:
                # EMA smoothing
                self.depth_threshold = 0.1 * current_threshold + 0.9 * self.depth_threshold
            
            # Create mask
            mask = (depth > self.depth_threshold).astype(np.float32)
            mask3 = mask[..., None]
            
            # Apply blur
            blurred = cv2.GaussianBlur(frame, self.blur_kernel, self.blur_sigma)
            result = (frame * mask3 + blurred * (1 - mask3)).astype(np.uint8)
            
            blurred_frames.append(result)
        
        # Recombine frames
        blurred_obs = np.concatenate(blurred_frames, axis=2)
        assert blurred_obs.shape == original_shape
        
        return blurred_obs
    
    def reset(self, **kwargs):
        """Reset depth threshold on episode reset."""
        self.depth_threshold = None
        return super().reset(**kwargs)


def create_env_from_df(
    hdf5_path,
    backgrounds_path,
    backgrounds_split,
    frame_stack=1,
    pixels_only=True,
    flatten_frames=True,
    difficulty=None,
    use_blur=False,
    blur_encoder='vits',
    blur_percentile=50,
    blur_kernel=(7, 7),
    blur_sigma=2.0,
    device='cuda',
):
    with h5py.File(hdf5_path, "r") as df:
        dm_env = suite.load(
            domain_name=df.attrs["domain_name"],
            task_name=df.attrs["task_name"],
            difficulty=df.attrs["difficulty"] if difficulty is None else difficulty,
            dynamic=df.attrs["dynamic"],
            background_dataset_path=backgrounds_path,
            background_dataset_videos=backgrounds_split,
            pixels_only=pixels_only,
            render_kwargs=dict(height=df.attrs["img_hw"], width=df.attrs["img_hw"]),
        )
        env = DmControlCompatibilityV0(dm_env)
        env = gym.wrappers.ClipAction(env)

        if pixels_only:
            env = SelectPixelsObsWrapper(env)

        if frame_stack > 1:
            env = gym.wrappers.FrameStackObservation(env, stack_size=frame_stack)
            if flatten_frames:
                env = FlattenStackedFrames(env)
        
        # Apply depth-based blurring if requested (after frame stacking)
        if use_blur:
            env = DepthBlurWrapper(
                env,
                encoder=blur_encoder,
                percentile=blur_percentile,
                blur_kernel=blur_kernel,
                blur_sigma=blur_sigma,
                device=device
            )

    return env

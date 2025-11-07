import math
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from typing import Optional

import cv2
import numpy as np
import pyrallis
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torch.utils.tensorboard import SummaryWriter
from pyrallis import field
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from tqdm import trange

# Add Depth-Anything-V2 to path
sys.path.insert(0, 'Depth-Anything-V2')
from depth_anything_v2.dpt import DepthAnythingV2

from src.augmentations import Augmenter
from src.nn import LAPO, ActionDecoder, Actor
from src.scheduler import linear_annealing_with_warmup
from src.utils import (
    DCSInMemoryDataset,
    DCSLAOMInMemoryDataset,
    create_env_from_df,
    get_grad_norm,
    get_optim_groups,
    normalize_img,
    set_seed,
    unnormalize_img,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class DepthBlurAugmenter:
    """Applies depth-based background blurring using DepthAnything V2."""
    
    def __init__(self, encoder='vitb', percentile=50, blur_kernel=(31, 31), blur_sigma=11, 
                 mask_kernel=(21, 21), mask_sigma=10, device='cuda'):
        """
        Args:
            encoder: DepthAnything encoder size ('vits', 'vitb', 'vitl', 'vitg')
            percentile: Depth percentile threshold for foreground/background separation
            blur_kernel: Kernel size for Gaussian blur on background
            blur_sigma: Sigma for Gaussian blur on background
            mask_kernel: Kernel size for mask feathering
            mask_sigma: Sigma for mask feathering
            device: Device to run depth model on
        """
        self.percentile = percentile
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.mask_kernel = mask_kernel
        self.mask_sigma = mask_sigma
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
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model = self.model.to(device).eval()
        
        print(f"Loaded DepthAnything V2 ({encoder}) for depth-based blurring")
    
    def apply_blur(self, img_tensor):
        """
        Apply depth-based blur to a batch of images.
        
        Args:
            img_tensor: Tensor of shape (B, C, H, W) in range [-1, 1]
        
        Returns:
            Blurred tensor of same shape
        """
        batch_size = img_tensor.shape[0]
        blurred_batch = []
        
        for i in range(batch_size):
            # Convert from tensor [-1, 1] to numpy uint8 [0, 255]
            img = img_tensor[i].cpu().numpy()
            img = ((img + 1) * 127.5).clip(0, 255).astype(np.uint8)
            img = img.transpose(1, 2, 0)  # CHW -> HWC
            
            # Infer depth
            with torch.no_grad():
                depth = self.model.infer_image(img)  # HxW depth map
            
            # Normalize depth to [0, 255]
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)
            
            # Create mask: foreground (high depth values) vs background (low depth values)
            p = np.percentile(depth, self.percentile)
            mask = (depth > p).astype(np.float32)
            
            # Feather mask edges
            mask = cv2.GaussianBlur(mask, self.mask_kernel, self.mask_sigma)
            mask3 = mask[..., None]  # Add channel dimension
            
            # Apply Gaussian blur to background
            blurred = cv2.GaussianBlur(img, self.blur_kernel, self.blur_sigma)
            
            # Composite: keep foreground sharp, blur background
            result = (img * mask3 + blurred * (1 - mask3)).astype(np.uint8)
            
            # Convert back to tensor [-1, 1]
            result = result.transpose(2, 0, 1)  # HWC -> CHW
            result = (result / 127.5 - 1.0).astype(np.float32)
            blurred_batch.append(torch.from_numpy(result))
        
        return torch.stack(blurred_batch).to(img_tensor.device)


@dataclass
class LAPOConfig:
    num_epochs: int = 100
    batch_size: int = 256
    future_obs_offset: int = 1
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    grad_norm: Optional[float] = None
    latent_action_dim: int = 8
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 1
    encoder_deep: bool = True
    frame_stack: int = 3
    data_path: str = "data/test.hdf5"
    # Blurring parameters
    use_blur: bool = True
    blur_prob: float = 0.5  # Probability of applying blur
    depth_encoder: str = 'vitb'  # 'vits', 'vitb', 'vitl', 'vitg'
    blur_percentile: int = 50  # Depth percentile for foreground/background


@dataclass
class BCConfig:
    num_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    encoder_scale: int = 1
    encoder_num_res_blocks: int = 2
    encoder_deep: bool = False
    dropout: float = 0.0
    use_aug: bool = True
    frame_stack: int = 3
    data_path: str = "data/test.hdf5"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 10
    eval_seed: int = 0


@dataclass
class DecoderConfig:
    total_updates: int = 1
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_epochs: int = 5
    hidden_dim: int = 128
    use_aug: bool = True
    data_path: str = "data/test.hdf5"
    dcs_backgrounds_path: str = "DAVIS/JPEGImages/480p"
    dcs_backgrounds_split: str = "train"
    eval_episodes: int = 10
    eval_seed: int = 0


@dataclass
class Config:
    project: str = "laom"
    group: str = "lapo-blur"
    name: str = "lapo-blur"
    seed: int = 0

    lapo: LAPOConfig = field(default_factory=LAPOConfig)
    bc: BCConfig = field(default_factory=BCConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)

    def __post_init__(self):
        self.name = f"{self.name}-{str(uuid.uuid4())}"


def train_lapo(config: LAPOConfig):
    dataset = DCSLAOMInMemoryDataset(
        config.data_path, max_offset=config.future_obs_offset, frame_stack=config.frame_stack, device=DEVICE
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    lapo = LAPO(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        latent_act_dim=config.latent_action_dim,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
    ).to(DEVICE)

    torchinfo.summary(
        lapo,
        input_size=[
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
            (1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        ],
    )
    optim = torch.optim.Adam(params=get_optim_groups(lapo, config.weight_decay), lr=config.learning_rate, fused=True)

    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    linear_probe = nn.Linear(config.latent_action_dim, dataset.act_dim).to(DEVICE)
    probe_optim = torch.optim.Adam(linear_probe.parameters(), lr=config.learning_rate)

    # Initialize depth-based blur augmenter if enabled
    blur_augmenter = None
    if config.use_blur:
        blur_augmenter = DepthBlurAugmenter(
            encoder=config.depth_encoder,
            percentile=config.blur_percentile,
            device=DEVICE
        )

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        lapo.train()
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, next_obs, future_obs, actions, _, _ = [b.to(DEVICE) for b in batch]
            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            next_obs = normalize_img(next_obs.permute((0, 3, 1, 2)))
            future_obs = normalize_img(future_obs.permute((0, 3, 1, 2)))

            # Apply depth-based blur augmentation with probability
            if blur_augmenter is not None and np.random.rand() < config.blur_prob:
                obs = blur_augmenter.apply_blur(obs)
                future_obs = blur_augmenter.apply_blur(future_obs)

            # update lapo
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_next_obs, latent_action = lapo(obs, future_obs)
                loss = F.mse_loss(pred_next_obs, next_obs)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            if config.grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(lapo.parameters(), max_norm=config.grad_norm)
            optim.step()
            scheduler.step()

            # update linear probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_action = linear_probe(latent_action.detach())
                probe_loss = F.mse_loss(pred_action, actions)

            probe_optim.zero_grad(set_to_none=True)
            probe_loss.backward()
            probe_optim.step()

            writer.add_scalar("lapo/mse_loss", loss.item(), total_steps)
            writer.add_scalar("lapo/action_probe_mse_loss", probe_loss.item(), total_steps)
            writer.add_scalar("lapo/throughput", total_tokens / (time.time() - start_time), total_steps)
            writer.add_scalar("lapo/learning_rate", scheduler.get_last_lr()[0], total_steps)
            writer.add_scalar("lapo/grad_norm", get_grad_norm(lapo).item(), total_steps)
            writer.add_scalar("lapo/epoch", epoch, total_steps)

        # logging reconstruction of next state
        obs_example = [unnormalize_img(next_obs[0][i : i + 3]) for i in range(0, 3 * config.frame_stack, 3)]
        next_obs_example = [unnormalize_img(pred_next_obs[0][i : i + 3]) for i in range(0, 3 * config.frame_stack, 3)]
        reconstruction_img = make_grid(obs_example + next_obs_example, nrow=config.frame_stack, padding=1)
        writer.add_image("lapo/next_obs_pred", reconstruction_img, total_tokens)

    return lapo


@torch.no_grad()
def evaluate_bc(env, actor, num_episodes, seed=0, device="cpu", action_decoder=None):
    returns = []
    for ep in trange(num_episodes, desc="Evaluating", leave=False):
        total_reward = 0.0
        obs, info = env.reset(seed=seed + ep)
        done = False
        while not done:
            obs_ = torch.tensor(obs.copy(), device=device)[None].permute(0, 3, 1, 2)
            obs_ = normalize_img(obs_)
            action, obs_emb = actor(obs_)
            if action_decoder is not None:
                if isinstance(action_decoder, ActionDecoder):
                    action = action_decoder(obs_emb, action)
                else:
                    action = action_decoder(action)

            obs, reward, terminated, truncated, info = env.step(action.squeeze().cpu().numpy())
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)

    return np.array(returns)


def train_bc(lam: LAPO, config: BCConfig):
    dataset = DCSInMemoryDataset(config.data_path, frame_stack=config.frame_stack, device=DEVICE)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )
    eval_env = create_env_from_df(
        config.data_path,
        config.dcs_backgrounds_path,
        config.dcs_backgrounds_split,
        frame_stack=config.frame_stack,
    )
    print(eval_env.observation_space)
    print(eval_env.action_space)

    num_actions = lam.latent_act_dim
    for p in lam.parameters():
        p.requires_grad_(False)
    lam.eval()

    actor = Actor(
        shape=(3 * config.frame_stack, dataset.img_hw, dataset.img_hw),
        num_actions=num_actions,
        encoder_scale=config.encoder_scale,
        encoder_channels=(16, 32, 64, 128, 256) if config.encoder_deep else (16, 32, 32),
        encoder_num_res_blocks=config.encoder_num_res_blocks,
        dropout=config.dropout,
    ).to(DEVICE)

    optim = torch.optim.AdamW(params=get_optim_groups(actor, config.weight_decay), lr=config.learning_rate, fused=True)
    # scheduler setup
    total_updates = len(dataloader) * config.num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    # for debug
    print("Latent action dim:", num_actions)
    act_decoder = nn.Sequential(
        nn.Linear(num_actions, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, dataset.act_dim)
    ).to(DEVICE)

    act_decoder_optim = torch.optim.AdamW(params=act_decoder.parameters(), lr=config.learning_rate, fused=True)
    act_decoder_scheduler = linear_annealing_with_warmup(act_decoder_optim, warmup_updates, total_updates)

    torchinfo.summary(actor, input_size=(1, 3 * config.frame_stack, dataset.img_hw, dataset.img_hw))
    if config.use_aug:
        augmenter = Augmenter(img_resolution=dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0
    for epoch in trange(config.num_epochs, desc="Epochs"):
        actor.train()
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, next_obs, true_actions = [b.to(DEVICE) for b in batch]
            # rescale from 0..255 -> -1..1
            obs = normalize_img(obs.permute((0, 3, 1, 2)))
            next_obs = normalize_img(next_obs.permute((0, 3, 1, 2)))

            # label with lapo latent actions
            target_actions = lam.label(obs, next_obs)

            # augment obs only for bc to make action labels determenistic
            if config.use_aug:
                obs = augmenter(obs)

            # update actor
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_actions, _ = actor(obs)
                loss = F.mse_loss(pred_actions, target_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            # optimizing the probe
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                pred_true_actions = act_decoder(pred_actions.detach())
                decoder_loss = F.mse_loss(pred_true_actions, true_actions)

            act_decoder_optim.zero_grad(set_to_none=True)
            decoder_loss.backward()
            act_decoder_optim.step()
            act_decoder_scheduler.step()

            writer.add_scalar("bc/mse_loss", loss.item(), total_steps)
            writer.add_scalar("bc/throughput", total_tokens / (time.time() - start_time), total_steps)
            writer.add_scalar("bc/learning_rate", scheduler.get_last_lr()[0], total_steps)
            writer.add_scalar("bc/act_decoder_probe_mse_loss", decoder_loss.item(), total_steps)
            writer.add_scalar("bc/epoch", epoch, total_steps)

    actor.eval()
    eval_returns = evaluate_bc(
        eval_env,
        actor,
        num_episodes=config.eval_episodes,
        seed=config.eval_seed,
        device=DEVICE,
        action_decoder=act_decoder,
    )
    writer.add_scalar("bc/eval_returns_mean", eval_returns.mean(), total_steps)
    writer.add_scalar("bc/eval_returns_std", eval_returns.std(), total_steps)

    return actor


def train_act_decoder(actor: Actor, config: DecoderConfig, bc_config: BCConfig):
    for p in actor.parameters():
        p.requires_grad_(False)
    actor.eval()

    dataset = DCSInMemoryDataset(config.data_path, frame_stack=bc_config.frame_stack, device=DEVICE)
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    # to make equal number of updates for all labeled datasets which vary in size
    num_epochs = config.total_updates // len(dataloader)

    action_decoder = ActionDecoder(
        obs_emb_dim=math.prod(actor.final_encoder_shape),
        latent_act_dim=actor.num_actions,
        true_act_dim=dataset.act_dim,
        hidden_dim=config.hidden_dim,
    ).to(DEVICE)

    optim = torch.optim.AdamW(
        params=get_optim_groups(action_decoder, config.weight_decay), lr=config.learning_rate, fused=True
    )
    eval_env = create_env_from_df(
        config.data_path,
        config.dcs_backgrounds_path,
        config.dcs_backgrounds_split,
        frame_stack=bc_config.frame_stack,
    )
    print(eval_env.observation_space)
    print(eval_env.action_space)

    # scheduler setup
    total_updates = len(dataloader) * num_epochs
    warmup_updates = len(dataloader) * config.warmup_epochs
    scheduler = linear_annealing_with_warmup(optim, warmup_updates, total_updates)

    if config.use_aug:
        augmenter = Augmenter(img_resolution=dataset.img_hw)

    start_time = time.time()
    total_tokens = 0
    total_steps = 0

    for epoch in trange(num_epochs, desc="Epochs"):
        for batch in dataloader:
            total_tokens += config.batch_size
            total_steps += 1

            obs, _, true_actions = [b.to(DEVICE) for b in batch]
            # rescale from 0..255 -> -1..1
            obs = normalize_img(obs.permute((0, 3, 1, 2)))

            if config.use_aug:
                obs = augmenter(obs)

            # update actor
            with torch.autocast(DEVICE, dtype=torch.bfloat16):
                with torch.no_grad():
                    latent_actions, obs_emb = actor(obs)
                pred_actions = action_decoder(obs_emb, latent_actions)

                loss = F.mse_loss(pred_actions, true_actions)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            scheduler.step()

            writer.add_scalar("decoder/mse_loss", loss.item(), total_steps)
            writer.add_scalar("decoder/throughput", total_tokens / (time.time() - start_time), total_steps)
            writer.add_scalar("decoder/learning_rate", scheduler.get_last_lr()[0], total_steps)
            writer.add_scalar("decoder/epoch", epoch, total_steps)

    actor.eval()
    eval_returns = evaluate_bc(
        eval_env,
        actor,
        num_episodes=config.eval_episodes,
        seed=config.eval_seed,
        device=DEVICE,
        action_decoder=action_decoder,
    )
    writer.add_scalar("decoder/eval_returns_mean", eval_returns.mean(), total_steps)
    writer.add_scalar("decoder/eval_returns_std", eval_returns.std(), total_steps)

    return action_decoder


@pyrallis.wrap()
def train(config: Config):
    log_dir = f"runs/{config.group}/{config.name}"
    global writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Log config
    config_dict = asdict(config)
    for key, value in config_dict.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                writer.add_text(f"config/{key}/{sub_key}", str(sub_value), 0)
        else:
            writer.add_text(f"config/{key}", str(value), 0)
    
    set_seed(config.seed)
    # stage 1: pretraining lapo on unlabeled dataset with depth-based blur
    lapo = train_lapo(config=config.lapo)
    # stage 2: pretraining bc on latent actions
    actor = train_bc(lam=lapo, config=config.bc)
    # stage 3: finetune on labeles ground-truth actions
    action_decoder = train_act_decoder(actor=actor, config=config.decoder, bc_config=config.bc)

    writer.close()
    return lapo, actor, action_decoder


if __name__ == "__main__":
    train()
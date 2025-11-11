# Latent Action Learning Requires Supervision in the Presence of Distractors

[[Project]](https://laom.dunnolab.ai/)
[[Paper]](https://arxiv.org/abs/2502.00379)
[[Twitter]](https://x.com/how_uhh/status/1927487077345841576)

Official implementation of the [**Latent Action Learning Requires Supervision in the Presence of Distractors**](https://arxiv.org/abs/2502.00379). Through empirical investigation, we demonstrate that supervision is necessary for good performance in latent action learning, highlighting a major limitation of current methods.

<img src="images/final_result_comb.jpg" alt="Environments" width="1000">

## Setup instructions

To set up python environment (with dev-tools of your taste, in our workflow we used conda and python 3.11), just install all the requirements:
```bash
pip install -r requirements.txt
```
For convienece we also provide the Dockerfile used in the experiments.

### Distracting Control Suite

We use slightly modified version of the original [Distracting Control Suite](https://arxiv.org/pdf/2101.02722), which we provide in the `src/dcs` for reproducibility. We changed difficulties and removed tensorflow from the dependencies, rewriting neccessary parts with numpy or PIL.  

You also need to get the DAVIS dataset, which is used for distracting backgrounds in the original DCS and in our work. We refer to the instructions in the [original repo](https://github.com/google-research/google-research/tree/master/distracting_control). You can put it wherever you like – all our scripts just need a path to it.

## Data 

<img src="images/envs-vis.png" alt="Environments" width="600">

### Downloading

We provide the exact datasets we used for the experiments. Each dataset is around 60GB (without labeled in the name) and consits of 5k trajectories, 1000 steps each (so 5M transitions in total). All datasets combined (for four envs, with and without distractors, and for ablations) are around 1.1TB, so be carefull. The links for datasets downloading from our s3 bucket are in the `data-links.txt`.

We provide small sample in the `data/example-data.hdf5` for convienece, just to demonstrate the format.

### Collecting from scratch

We provide scripts (and checkpoints) used for datasets collection in `scripts/data_collection`. 

We pre-trained expert policies with PPO for `cheetah-run`, `walker-run` and `hopper-hop`. PPO was adapted from beautiful [CleanRL](https://github.com/vwxyzjn/cleanrl) library. See `scripts/data_collection/collection/cleanrl_ppo.py`. We used almost default hyperparameters and trained for 1_000_000_000 transitions. Example wandb runs: [cheetah-run logs](https://wandb.ai/state-machine/lapo/runs/2a1dfdha), [walker-run logs](https://wandb.ai/state-machine/lapo/runs/t6xlpt7v), [hopper-hop logs](https://wandb.ai/state-machine/lapo/runs/6ejbglhv). You can find the exact hyperparameters in Overview->Config. Unfortunately, we were unable to get satisfactory performance on `humanoid-walk` with PPO, so instead we used SAC from [stable-baselines3](https://github.com/DLR-RM/stable-baselines3) library with default hyperparameters and trained for 2_000_000 transitions. See `scripts/data_collection/sb3_sac.py`. 

All experts were pre-trained with proprioceptive observations. We render images only during data collection. Returnes of the experts we provide (in checkpoints and the datasets):

| Dataset        | Average Return |
|----------------|----------------|
| cheetah-run    | 837.70         |
| walker-run     | 739.79         |
| hopper-hop     | 306.63         |
| humanoid-walk  | 617.22         |

With checkpoints available, data in required format can be collected with following scripts:
```bash
# example for ppo checkpoints
# use dcs_difficulty=vanilla to collectd data without distractors
python -m scripts.data_collection.collect_data \
    --checkpoint_path="scripts/data_collection/checkpoints/hopper-hop-expert" \
    --checkpoint_name="checkpoint.pt" \
    --dcs_backgrounds_path="DAVIS/JPEGImages/480p" \
    --save_path="data/hopper-hop-test.hdf5" \
    --num_trajectories=5 \
    --dcs_difficulty="scale_easy_video_hard" \  
    --dcs_backgrounds_split="train" \
    --dcs_img_hw=64 \
    --seed=0 \
    --cuda=False

# example for sac checkpoints
python -m scripts.data_collection.collect_data_sb3 \
    --checkpoint_path="scripts/data_collection/checkpoints/sac-humanoid-walk" \
    --dcs_backgrounds_path="DAVIS/JPEGImages/480p" \
    --save_path="data/humanoid-walk-test.hdf5" \
    --num_trajectories=10 \
    --dcs_difficulty="scale_easy_video_hard" \
    --dcs_backgrounds_split="train" \
    --dcs_img_hw=64 \
    --seed=0 \
    --cuda=False
```

To simulate access to small datasets with ground-truth action labels we used simple script which samples trajectories from full dataset, see `scripts/sample_labeled_data.py`. However, you can also collect these with scripts for collection above, there is no real difference.
```bash
python -m scripts.sample_labeled_data \
    --data_path="path/to/full/dataset" \
    --save_path="path/to/full/dataset-labeled-1000x$num_traj.hdf5" \
    --chunk_size=1000 \
    --num_trajectories=$num_traj
```

## Data Processing & Visualization Scripts

We provide several utility scripts for data processing and visualization:

### Blurring Background

Apply depth-based background blurring to datasets (useful for training with reduced background distractions):

```bash
# Blur observations using depth estimation
python scripts/blur_hdf5_data.py \
    --input data/example-data.hdf5 \
    --output data/example-data-blurred.hdf5 \
    --encoder vits \
    --percentile 30 \
    --blur-kernel 7 7 \
    --blur-sigma 2.0
```

Parameters:
- `--encoder`: DepthAnything model size (`vits`, `vitb`, `vitl`, `vitg`) - smaller is faster
- `--percentile`: Depth threshold for foreground/background (30 = more foreground kept sharp)
- `--blur-kernel`: Gaussian blur kernel size (smaller = gentler blur)
- `--blur-sigma`: Blur strength (smaller = more visible background)

### Generating Foreground Masks

Pre-generate depth-based masks for mask-based training (saves masks separately to keep files small):

```bash
# Generate masks and save to separate file
python scripts/generate_masks_hdf5.py \
    --input data/example-data.hdf5 \
    --output data/example-data-masks.hdf5 \
    --encoder vits \
    --percentile 50
```

The masks are stored as single-channel uint8 (much smaller than images) and loaded automatically during training.

### Visualizing Trajectories

#### Interactive Playback

```bash
# Play a single trajectory
python scripts/visualize_trajectory.py \
    --input data/example-data.hdf5 \
    --trajectory 0 \
    --fps 30

# Compare original vs blurred side-by-side
python scripts/visualize_trajectory.py \
    --input data/example-data.hdf5 \
    --input-blurred data/example-data-blurred.hdf5 \
    --trajectory 0
```

Controls: `q` to quit, `p` to pause/resume, `r` to restart


## Running experiments

We provide training scripts for all methods from the paper: IDM, LAPO, LAOM, LAOM+supervision. For clarity and educational purposes, all scrips are single-file and implement all stages of the LAM pipline at once: latent action model pre-training, behavioral cloning, action decoder fine-tuning. 

<img src="images/lapo-pipeline.jpg" alt="Environments" width="600">

> [!NOTE] 
> **WARN**: This is not the most efficient implementation if you need to run a lot of experiments with different hyperparameters, as it will waste time re-training from scratch duplicate parts of the pipeline. In such a case, it would be better to split these scripts into several modular ones (one for each stage).

We provide the configs used in the experiments in the `configs`. You only need to provide all the paths to the required datasets:

### LAPO (Latent Action Pre-training from Observations)

```bash
python -m train_lapo \
    --config_path="configs/lapo.yaml" \
    --lapo.data_path="data/example-data.hdf5" \
    --bc.data_path="data/example-data.hdf5" \
    --bc.dcs_backgrounds_path="DAVIS/JPEGImages/480p" \
    --decoder.data_path="data/example-data.hdf5" \
    --decoder.dcs_backgrounds_path="DAVIS/JPEGImages/480p"
```

For training with blurred data (and possible blurred eval):
```bash
# First, blur the dataset
python scripts/blur_hdf5_data.py \
    --input data/example-data.hdf5 \
    --output data/example-data-blurred.hdf5

# Then train with blurred data and enable evaluation-time blurring
python -m train_lapo \
    --config_path="configs/lapo.yaml" \
    --lapo.data_path="data/example-data-blurred.hdf5" \
    --bc.data_path="data/example-data-blurred.hdf5" \
    --bc.eval_use_blur=True \
    --bc.dcs_backgrounds_path="DAVIS/JPEGImages/480p" \
    --decoder.data_path="data/example-data-blurred.hdf5" \
    --decoder.eval_use_blur=True \
    --decoder.dcs_backgrounds_path="DAVIS/JPEGImages/480p"
```

For training with masked loss:
```bash
# First, generate masks
python scripts/generate_masks_hdf5.py \
    --input data/example-data.hdf5 \
    --output data/example-data-masks.hdf5

# Then train with mask-based loss
python -m train_lapo_mask \
    --config_path="configs/lapo-mask.yaml" \
    --lapo.data_path="data/example-data.hdf5" \
    --lapo.masks_path="data/example-data-masks.hdf5" \
    --bc.data_path="data/example-data.hdf5" \
    --bc.dcs_backgrounds_path="DAVIS/JPEGImages/480p" \
    --decoder.data_path="data/example-data.hdf5" \
    --decoder.dcs_backgrounds_path="DAVIS/JPEGImages/480p"
```

### LAOM with Supervision

```bash
python -m train_laom_labels \
    --config_path="configs/laom-labels.yaml" \
    --lapo.data_path="data/example-data.hdf5" \
    --lapo.labeled_data_path="data/example-data.hdf5" \
    --lapo.eval_data_path="data/example-data.hdf5" \
    --bc.data_path="data/example-data.hdf5" \
    --bc.dcs_backgrounds_path="DAVIS/JPEGImages/480p" \
    --decoder.dcs_backgrounds_path="DAVIS/JPEGImages/480p"
```

## Reproducing figures

For reproducibility purposes, we provide jupyter notebook which can reproduce all main figures from the paper based on our wandb logs (which are public).

See `scripts/reproducing_figures.ipynb`.

## Citing

```
@article{nikulin2025latent,
  title={Latent Action Learning Requires Supervision in the Presence of Distractors},
  author={Nikulin, Alexander and Zisman, Ilya and Tarasov, Denis and Lyubaykin, Nikita and Polubarov, Andrei and Kiselev, Igor and Kurenkov, Vladislav},
  journal={arXiv preprint arXiv:2502.00379},
  year={2025}
}
```

## Acknowledgments

This work was supported by [Artificial Intelligence Research Institute](https://airi.net/?force=en) (AIRI).

<img src="images/logo.png" align="center" width="20%" style="margin:15px;">


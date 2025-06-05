# MF-PBT: Multiple-Frequencies Population-Based Training

[![arXiv](https://img.shields.io/badge/arXiv-2506.03225-b31b1b.svg)](https://arxiv.org/abs/2506.03225)

This repository contains the official implementation of **Multiple-Frequencies Population-Based Training (MF-PBT)**, a novel hyperparameter optimization algorithm that addresses the greediness issues in standard Population-Based Training (PBT).

**Technical Notes**: This codebase uses the [Brax](https://github.com/google/brax) library for continuous control environments and adapts PPO implementation from Brax's repository. [JAX](https://github.com/google/jax) is used to efficiently manage populations of agents in parallel across multiple devices. The code will automatically utilize all available GPUs - use `CUDA_VISIBLE_DEVICES` to control GPU usage.

## Supported Algorithms

- **MF-PBT** (`mfpbt`): Our proposed Multiple-Frequencies Population-Based Training
- **PBT** (`pbt`): Standard Population-Based Training baseline
- **PB2** (`pb2`): Population-Based Bandits baseline, adapted from [this repo](https://github.com/jparkerholder/PB2)
- **Random Search** (`random_search`): Random search baseline
- **Ablation** (`ablation`): Ablation on the asymmetric migration, corresponds to section 5.2 of the paper
- **Do Nothing** (`do_nothing`): Control baseline, used in section 4.3 of the paper

## Installation

Tested on Python 3.11

Please refer to the [Jax installation guide](https://jax.readthedocs.io/en/latest/installation.html) for detailed instructions on installing JAX with GPU support.

```bash
# Clone the repository
git clone <repository-url>
cd MF-PBT

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Example: MF-PBT on HalfCheetah

```bash
python main.py --env halfcheetah --hpo mfpbt --exp-name my_experiment --num-agents 32
```

### Using Configuration Files

You can also specify a custom configuration file:

```bash
python main.py --config-file configurations/halfcheetah/mfpbt.yml --exp-name my_experiment
```

### Variance-Exploitation Mode

Run algorithms in variance-exploitation mode where hyperparameters remain fixed but PBT is used for model selection:

```bash
python main.py --env halfcheetah --hpo mfpbt --variance-exploitation --num-agents 32
```

## Usage

### Basic Command Structure

```bash
# Using automatic config selection
python main.py --env ENV_NAME --hpo ALGORITHM [options]

# Using specific config file
python main.py --config-file CONFIG_PATH [options]

# Running with default example (hopper + mfpbt)
python main.py [options]
```

### Key Arguments

**Core Arguments:**
- `--env`: Environment (`ant`, `halfcheetah`, `hopper`, `humanoid`, `pusher`, `walker2d`)
- `--hpo`: Algorithm (`mfpbt`, `pbt`, `pb2`, `random_search`, `ablation`, `do_nothing`)
- `--config-file`: Path to custom configuration file (overrides `--env`/`--hpo` selection)

**Experiment Control:**
- `--exp-name`: Experiment name for logging (auto-generated if not provided)
- `--variance-exploitation`: Enable variance-exploitation mode
- `--num-agents`: Number of agents in population
- `--num-rounds`: Number of training rounds
- `--num-timesteps-round`: Environment steps per round ($t_{\mathrm{ready}}$ parameter from paper)
- `--num-envs-per-agent`: Number of Brax environments per agent

**MF-PBT Specific:**
- `--frequencies`: Evolution frequencies for MF-PBT (e.g., `--frequencies 1 10 25 50`)

**Logging & Seeds:**
- `--logging-directory`: Custom logging directory (defaults to `runs/` in project directory)
- `--jax-seed`: JAX seed for training and environment simulations (default: 0)
- `--numpy-seed`: NumPy seed for hyperparameter initialization (default: 0)

### Important Parameters

**$t_{\mathrm{ready}}$ (--num-timesteps-round)**: This is the crucial "ready time" parameter from the paper that determines how many environment steps each agent performs before the next evolution/exploitation step. This directly affects the trade-off between exploration and exploitation in the population-based training process.

### MF-PBT Requirements

**Important**: For MF-PBT to work properly:
- The number of agents must be divisible by the number of frequencies
- The number of agents per population (num_agents ÷ num_frequencies) must be divisible by 4

For example, with `--frequencies 1 10 25 50` (4 frequencies) and `--num-agents 32`, you get 8 agents per population, which satisfies both requirements.

## Configuration Files

Algorithm and environment settings are defined in `configurations/ENV_NAME/ALGORITHM.yml`. **Note**: Core hyperparameter search spaces and PBT exploration/exploitation parameters must be specified in the YAML files and cannot be overridden via command line.

**See `configurations/example.yml` for a detailed example configuration with comprehensive comments explaining all options.**

### Hyperparameters Being Optimized

By default, the experiments optimize two key PPO hyperparameters:
- **Learning Rate**: Optimized in log-space (10^-5 to 10^-3)
- **Entropy Cost**: Optimized in log-space (10^-3 to 10^-1)

These search spaces are defined in the configuration files for each environment. If you wish to optimize additional or different hyperparameters, simply modify the corresponding YAML configuration files.

### Customizing Exploitation and Exploration

Various exploitation and exploration functions can be implemented or customized in `hpo/utils`. To experiment with different strategies, edit or add new functions in this module and reference them in your configuration files.

## Logging and Results

### Default Logging

By default, experiments are logged locally in the `runs/` directory:

```
runs/
└── {exp_name}/
    └── logs/
        ├── round_0_logs.csv
        ├── round_0_best_model.pkl
        ├── round_1_logs.csv
        ├── round_1_best_model.pkl
        └── ...
```

### Experiment Name Generation

If no `--exp-name` is provided, the system auto-generates names using the pattern:
```
{algorithm}/{environment}/n_agents_{num_agents}
```

For example: `mfpbt/halfcheetah/n_agents_32`

With variance-exploitation mode: `mfpbt/halfcheetah/n_agents_32_variance_exploit`

### Custom Logging Directory

To log experiments to a custom directory (e.g., network storage, cloud mount):

```bash
python main.py --env halfcheetah --hpo mfpbt --logging-directory /path/to/custom/logs
```

This will create logs at `/path/to/custom/logs/{exp_name}/logs/`

### CSV Log Format

Each `round_X_logs.csv` file contains detailed information for all agents at round X:

| Column | Description |
|--------|-------------|
| `index` | Agent index (0 to num_agents-1) |
| `parent` | Index of parent agent (for tracking evolution lineage) |
| `reward` | Episode reward achieved by the agent |
| `env_steps` | Total environment steps completed by this agent |
| `entropy_cost` | Current entropy cost hyperparameter value |
| `learning_rate` | Current learning rate hyperparameter value |

**Example CSV content:**
```csv
index,parent,reward,env_steps,entropy_cost,learning_rate
0,2,1083.5088,3604480,0.0012076665792328134,0.000398943692974848
1,1,1086.5841,3604480,0.001886979030051271,0.00016340733664249772
2,2,1096.6401,3604480,0.0015095832240410167,0.0003191549543798784
3,3,1086.7576,3604480,0.0015095832240410167,0.00020425917080312214
```

### Resuming Experiments

To resume an interrupted experiment:
```bash
# Simply rerun with the same experiment name and logging-directory (if used)
python main.py --env halfcheetah --hpo mfpbt --exp-name my_experiment --num-agents 32
```

The system automatically detects existing logs and resumes from the last completed round.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@misc{doulazmi2025mfpbt,
title={Multiple-Frequencies Population-Based Training},
author={Waël Doulazmi and Auguste Lehuger and Marin Toromanoff and Valentin Charraut and Thibault Buhet and Fabien Moutarde},
year={2025},
eprint={2506.03225},
archivePrefix={arXiv},
primaryClass={cs.LG},
url={https://arxiv.org/abs/2506.03225},
}
```

## License

MIT License


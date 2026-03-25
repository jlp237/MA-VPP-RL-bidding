# VPP RL Bidding

RL-based strategic bidding for Virtual Power Plants across multiple electricity markets.

## Overview

This project implements a Gymnasium reinforcement learning environment where an RL agent learns to optimize bidding strategies for a Virtual Power Plant (VPP) across multiple electricity markets. The VPP aggregates distributed generation assets (hydro, wind, solar) and must decide how to allocate capacity and at what price to bid in each market.

**Origin**: Based on the master's thesis *"Reinforcement Learning based Strategic Bidding in the Balancing Market with a Virtual Power Plant"*, restructured into a professional, extensible Python project.

## Supported Markets

| Market | Status | Description |
|---|---|---|
| **FCR** (Frequency Containment Reserve) | Implemented | Sealed-bid capacity tender (Primärregelleistung), 6x4h blocks/day, gate closure D-1 08:45 CET, pay-as-cleared |
| **Day-Ahead** (EPEX Spot DA-Auktion) | Planned | Hourly energy auction, 24 products per day, gate closure D-1 12:00 CET |
| **aFRR** (Sekundärregelleistung) | Planned | Capacity + energy, 4h blocks |
| **mFRR** (Minutenreserve) | Planned | Manual reserve market |
| **Intraday Continuous** (EPEX/XBID) | Planned | Continuous order-book trading via XBID (European cross-border coupling) |
| **Imbalance** (reBAP / Ausgleichsenergiepreis) | Planned | Real-time settlement price exposure |

**Cross-market constraint**: Capacity committed to one market is unavailable for others. The agent must learn to split VPP capacity optimally across enabled markets.

## Quick Start

```bash
# Install dependencies
uv sync

# Train a PPO agent on the FCR market
uv run vpp-bidding train --algorithm PPO --timesteps 100000

# Evaluate a trained model
uv run vpp-bidding evaluate --model models/ppo_fcr.zip --episodes 70

# Hyperparameter tuning with Optuna
uv run vpp-bidding tune --algorithm PPO --n-trials 100
```

## Architecture

```
Agent Action (normalized [-1,1])
    │
    ▼
┌─────────────────────────────┐
│   VPP Environment           │
│                             │
│  ┌──────────┐ ┌──────────┐  │
│  │ FCR      │ │ Day-Ahead│  │   Each market implements
│  │ Market   │ │ Market   │  │   the Market ABC:
│  │ (impl.)  │ │ (stub)   │  │   - simulate()
│  └────┬─────┘ └────┬─────┘  │   - calculate_reward()
│       │             │       │   - get_capacity_commitment()
│       ▼             ▼       │
│  Capacity Budget Tracker    │   Enforces cross-market
│  (allocate → commit → next) │   capacity constraints
│       │                     │
│       ▼                     │
│  Aggregated Reward          │
└─────────────────────────────┘
    │
    ▼
Agent Observation (Dict space)
```

### Gate Closure Sequence

```
FCR tender:  D-1 08:45 CET
aFRR tender: D-1 10:45 CET
mFRR tender: D-1 11:30 CET
Day-Ahead:   D-1 12:00 CET
Intraday:    D-1 15:00 CET onwards (continuous)
Imbalance:   Real-time settlement
```

## Adding a New Market

1. Create `src/vpp_bidding/markets/your_market/market.py`
2. Implement the `Market` abstract base class from `markets/base.py`
3. Register it in `markets/registry.py`
4. Add a data loader in `data/loaders.py`
5. Enable it in config: `markets.enabled = ["fcr", "your_market"]`

```python
from vpp_bidding.markets.base import Market, MarketState

class YourMarket(Market):
    @property
    def name(self) -> str:
        return "your_market"

    @property
    def action_size(self) -> int:
        return 24  # e.g., 24 hourly bids

    def simulate(self, actions, data, step) -> MarketState:
        # Your clearing logic here
        ...
```

## Adding New Data Sources

1. Add fetching logic to `src/vpp_bidding/data/collect.py`
2. Add preprocessing to `src/vpp_bidding/data/preprocess.py`
3. Register the CSV path in your config TOML under `[data]`
4. Load it in your market's observation builder

## RL Algorithms

8 algorithms are supported via stable-baselines3:

| Algorithm | Type | Library |
|---|---|---|
| PPO | On-policy | stable-baselines3 |
| A2C | On-policy | stable-baselines3 |
| TRPO | On-policy | sb3-contrib |
| RecurrentPPO | On-policy (LSTM) | sb3-contrib |
| SAC | Off-policy | stable-baselines3 |
| DDPG | Off-policy | stable-baselines3 |
| TD3 | Off-policy | stable-baselines3 |
| TQC | Off-policy | sb3-contrib |

## CLI Commands

```bash
vpp-bidding train      --config configs/vpp_1_training.toml --algorithm PPO --timesteps 100000
vpp-bidding evaluate   --model models/ppo.zip --config configs/vpp_2.toml --episodes 70
vpp-bidding tune       --config configs/vpp_1_tuning.toml --algorithm PPO --n-trials 100
vpp-bidding collect-data --output-dir data/raw
```

## Project Structure

```
├── pyproject.toml              # Project config (uv, ruff, pytest, mypy)
├── configs/                    # TOML configuration files
│   ├── vpp_1_training.toml     # Training config (3 hydro x 10MW)
│   ├── vpp_1_tuning.toml       # Tuning config (validation set)
│   ├── vpp_2.toml              # Test config (70 days)
│   ├── vpp_4.toml              # Large VPP test (288 units, 140 days)
│   └── tuning/optuna.toml      # Optuna search spaces
├── data/
│   ├── raw/                    # Raw data from regelleistung.net, SimBench, SMARD
│   └── clean/                  # Preprocessed CSVs for the environment
├── src/vpp_bidding/
│   ├── cli.py                  # CLI entry points
│   ├── config.py               # TOML config loading + dataclasses
│   ├── domain/                 # Enums, constants, dataclasses
│   ├── markets/
│   │   ├── base.py             # Abstract Market class
│   │   ├── registry.py         # Market discovery + instantiation
│   │   ├── fcr/                # FCR market (implemented)
│   │   ├── day_ahead/          # Day-ahead market (stub)
│   │   ├── afrr/               # aFRR market (stub)
│   │   ├── mfrr/               # mFRR market (stub)
│   │   ├── intraday/           # Intraday market (stub)
│   │   └── imbalance/          # Imbalance market (stub)
│   ├── env/
│   │   ├── vpp_env.py          # Main Gymnasium environment
│   │   ├── vpp.py              # VPP asset configuration
│   │   ├── render.py           # Plotly + WandB visualization
│   │   └── registration.py     # Gymnasium env registration
│   ├── training/
│   │   ├── algorithms.py       # Algorithm factory (8 algos)
│   │   ├── train.py            # Training loop
│   │   ├── evaluate.py         # Model evaluation
│   │   ├── tuning.py           # Optuna hyperparameter tuning
│   │   └── callbacks.py        # WandB callbacks
│   └── data/
│       ├── collect.py          # Data collection pipeline
│       ├── preprocess.py       # Data cleaning
│       └── loaders.py          # CSV loading utilities
└── tests/                      # Comprehensive test suite
```

## Development

```bash
# Install with dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Lint
uv run ruff check src/ tests/

# Type check
uv run mypy src/

# Format
uv run ruff format src/ tests/
```

## Configuration

Configuration uses TOML files. Key sections:

```toml
[markets]
enabled = ["fcr"]  # Add markets as they're implemented

[time]
first_slot_date_start = "2020-07-02 22:00:00+00:00"
last_slot_date_end = "2022-05-31 21:45:00+00:00"

[data]
renewables = "data/clean/renewables.csv"
# ... other data paths

[[assets.hydro]]
type = "run-of-river"
max_capacity_mw = 10.0
quantity = 1
max_fcr_capacity_share = 0.5
asset_column_names = ["Hydro1"]

[wandb]
project = "vpp-bidding"
mode = "online"
```

## Data Sources

- **regelleistung.net**: FCR tender data, auction results, anonymous bids (2020-2022)
- **SMARD (smard.de)**: Wholesale electricity prices
- **SimBench**: Renewable generation profiles (hydro, wind, solar)

## Not Implemented (Out of Scope)

- **Redispatch market** (Redispatch 2.0, TSO congestion management)
- **Capacity mechanism** (medium-term investment commitments)
- **Renewable curtailment** (Einspeisemanagement)
- **RR** (Replacement Reserve)

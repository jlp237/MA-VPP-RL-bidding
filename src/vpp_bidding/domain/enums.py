from __future__ import annotations

from enum import IntEnum, StrEnum


class SlotStatus(IntEnum):
    LOST = -1
    NOT_PARTICIPATED = 0
    WON = 1


class EnvMode(StrEnum):
    TRAINING = "training"
    EVAL = "eval"
    TEST = "test"


class RenderMode(StrEnum):
    HUMAN = "human"
    FAST_TRAINING = "fast_training"


class MarketType(StrEnum):
    DAY_AHEAD = "day_ahead"
    FCR = "fcr"
    AFRR_POWER = "afrr_power"
    AFRR_ENERGY = "afrr_energy"
    MFRR = "mfrr"
    INTRADAY = "intraday"
    IMBALANCE = "imbalance"


class Algorithm(StrEnum):
    PPO = "PPO"
    A2C = "A2C"
    TRPO = "TRPO"
    RECURRENT_PPO = "RecurrentPPO"
    SAC = "SAC"
    DDPG = "DDPG"
    TD3 = "TD3"
    TQC = "TQC"

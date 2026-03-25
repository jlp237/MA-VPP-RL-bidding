"""FCR market constants for the German Primaerregelleistung.

Product: Symmetric FCR capacity, 6 blocks x 4 hours per day.
Tender: Sealed-bid, day-ahead, gate closure D-1 08:45 CET.
Settlement: Pay-as-cleared (single settlement price per control area).
Data period: 2020-07-02 to 2022-05-31.

Activation: 30-second full activation requirement.
Reservation: Symmetric -- VPP must hold capacity for both positive
(increase output) and negative (decrease output) FCR simultaneously.
"""

# Time structure
SLOTS_PER_DAY: int = 6
STEPS_PER_SLOT: int = 16
STEPS_PER_DAY: int = 96

# Energy parameters
# Duration of each FCR activation check step (3.75 min = 0.0625h).
# Based on: 7.5 activations * 30s per 15-min interval, per positive/negative direction.
ACTIVATION_STEP_DURATION_HOURS: float = 0.0625
ENERGY_PER_ACTIVATION_STEP_HOURS = ACTIVATION_STEP_DURATION_HOURS  # backward compat alias

# Duration of each reservation check step in hours (15 min = 0.25h)
RESERVATION_STEP_DURATION_HOURS: float = 0.25
RESERVATION_ENERGY_HOURS = RESERVATION_STEP_DURATION_HOURS  # backward compat alias

# Activation parameters
# Probability of successful delivery per timestep. Based on Seidel & Haase VPP reliability model.
ACTIVATION_SUCCESS_PROBABILITY: float = 0.995

# Empirical FCR activation share distribution. Based on thesis analysis of German FCR
# activation patterns (2020-2022). 67.5% of activations require only 0-5% of contracted capacity.
ACTIVATION_POPULATION: list[float] = [0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1.0]
ACTIVATION_WEIGHTS: list[float] = [
    0.675,
    0.23,
    0.06,
    0.02,
    0.01,
    0.002,
    0.002,
    0.001,
]  # Weights sum to 1.0

# Penalty parameters
# Simplified penalty model (not actual TSO penalty rules). Based on thesis approximation:
# penalty = max(market_price * 1.25, market_price + 10 EUR, settlement_price) * undelivered_energy.
# Actual German TSO penalties vary by TSO and year.
PENALTY_SURCHARGE_FACTOR: float = 1.25
PENALTY_SURCHARGE_EUR: float = 10.0

# Reward parameters
REWARD_DISTANCE_EXPONENT: float = 0.2

# Price caps
# Historical maximum FCR settlement price observed in German system during 2020-2022 data period.
# Used for MinMaxScaler normalization only.
# Unit: EUR/MW per 4-hour block (capacity price, not energy price)
MAX_FCR_SETTLEMENT_PRICE_EUR_PER_MW: float = 4257.07
MAX_SETTLEMENT_PRICE_EUR_MW = MAX_FCR_SETTLEMENT_PRICE_EUR_PER_MW  # backward compat alias

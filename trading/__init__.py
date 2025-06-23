from gymnasium.envs.registration import register

register(
    id="trading/Trading-v0",
    entry_point="trading.envs:TradingEnv",
)

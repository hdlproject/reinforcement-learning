from gymnasium.envs.registration import register

register(
    id="line-following/LineFollowing-v0",
    entry_point="line-following.envs:LineFollowingEnv",
)

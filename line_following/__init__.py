from gymnasium.envs.registration import register

register(
    id="line_following/LineFollowing-v0",
    entry_point="line_following.envs:LineFollowingEnv",
)

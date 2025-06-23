from line_following.envs.line_following import LineFollowingEnv

def run():
    # initialise the environment
    env = LineFollowingEnv(render_mode='chart', sensor_number=5)

    # reset the environment to generate the first observation
    observation, info = env.reset()
    for _ in range(1000):
        # this is where you would insert your policy
        action = env.action_space.sample()

        # step (transition) through the environment with the action
        # receiving the next observation, reward and if the episode has terminated or truncated
        observation, reward, terminated, truncated, info = env.step(action)

        # if the episode has ended then we can reset to start a new episode
        if terminated or truncated:
            observation, info = env.reset()

    env.close()

import gymnasium as gym
import pygame
from gymnasium import spaces
import numpy as np
import math


class LineFollowingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, sensor_number=5, motor_number=2):
        self.sensor_number = sensor_number
        self.motor_number = motor_number

        # straight along the center line (y-axis)
        self.current_angle = 0
        self._x_pos = 0

        # calculate left and right border of x-axis
        if self.sensor_number % 2 == 1:
            self.max_x = math.floor(self.sensor_number / 2)
        else:
            self.max_x = (self.sensor_number / 2) - 1
        self.min_x = -1 * self.max_x

        self._agent_location = np.zeros(self.sensor_number)
        self._center_location = np.zeros(self.sensor_number)
        if self.sensor_number % 2 == 1:
            # if the sensor number is odd, the center position is indicated by
            # the middle sensor having the max sensor value
            self._center_location[math.floor(self.sensor_number / 2)] = 1.0
        else:
            # if the sensor number is even, the center position is indicated by
            # the two middle sensor having the max sensor value / 2
            self._center_location[self.sensor_number / 2] = 0.5
            self._center_location[(self.sensor_number / 2) - 1] = 0.5
        self._target_location = self._center_location

        self._farthest_location = np.zeros(self.sensor_number)
        self._farthest_location[self.sensor_number - 1] = 1
        self._farthest_distance = self._get_distance(self._center_location, self._farthest_location)

        self.observation_space = spaces.Dict(
            {
                # the value of a sensor is ranged between 0-1
                # the shape represent the size of the sensor grid
                "agent": spaces.Box(low=0.0, high=1.0, shape=(sensor_number,)),
                "target": spaces.Box(low=0.0, high=1.0, shape=(sensor_number,)),
            }
        )

        # the speed of right and left motors
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(motor_number,))

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        # The size of the PyGame window
        self.window_size = 512

    def _get_obs(self):
        return {
            "agent": self._agent_location,
            "target": self._target_location
        }

    def _get_info(self):
        return {
            "distance": self._get_distance(self._agent_location, self._target_location)
        }

    def _get_distance(self, loc1, loc2):
        return np.linalg.norm(loc1 - loc2, ord=2)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self._x_pos = np.random.randint(self.min_x, self.max_x)
        # self._agent_location = np.random.uniform(low=0.0, high=1.0, size=(self.sensor_number,))
        self._agent_location = self._convert_x_to_sensor_value(self._x_pos)
        self._target_location = self._center_location

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _simulate_next_location(self, left_speed, right_speed):
        # distance between wheels
        wheel_base = 1.0
        # forward speed
        v = (left_speed + right_speed) / 2.0
        # angular velocity
        # positive means left, negative means right
        omega = (right_speed - left_speed) / wheel_base

        # calculate new position
        self._x_pos += v * np.cos(self.current_angle)
        self.current_angle += omega

        return self._convert_x_to_sensor_value(self._x_pos)

    def _convert_x_to_sensor_value(self, x):
        # array of sensor value weight
        # e.g [-2, -1, 0, 1, 2]
        # e.g [-2, -1, 0, 0, 1, 2]
        sensor_pos = np.linspace(self.min_x, self.max_x, self.sensor_number)
        # get the sensor value weight
        # based on the real position
        # e.g if x=1, [-1, 0, 1, 2, 3]
        real_sensor_pos = sensor_pos + x
        # simulate next sensor reading with Gaussian formula
        # e.g if x=1, [0.1353, 1.0, 0.1353, 0.000335, 0.0000000152]
        sensor_sensitivity = 0.5
        next_sensor_reading = np.exp(-(real_sensor_pos ** 2) / (2 * sensor_sensitivity ** 2))
        # normalize the value to 0.0 - 1.0
        # e.g if x=1, [0.1353, 1.0, 0.1353, 0.000335, 0.0000000152]
        return next_sensor_reading / np.max(next_sensor_reading)

    def step(self, action):
        left_speed, right_speed = action

        self._agent_location = self._simulate_next_location(left_speed, right_speed)
        observation = self._get_obs()
        info = self._get_info()

        terminated = np.array_equal(self._agent_location, self._target_location)

        reward = info["distance"] / self._farthest_distance

        return observation, reward, terminated, False, info

    def render(self):
        self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # +2 to represent totally off the track position
        horizontal_square = (self.sensor_number + 2)
        pix_square_size = self.window_size / horizontal_square

        # draw center line
        y_center_line = (horizontal_square / 2) - (pix_square_size / 2)
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                (0, y_center_line),
                (pix_square_size, self.window_size),
            )
        )

        for x in range(horizontal_square):
            pygame.draw.line(
                canvas,
                0,
                (0, pix_square_size * x),
                (self.window_size, pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                canvas,
                0,
                (pix_square_size * x, 0),
                (pix_square_size * x, self.window_size),
                width=3,
            )

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()

            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)),
                axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

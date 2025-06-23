import gymnasium as gym
import pygame
from gymnasium import spaces
import numpy as np
import math
import matplotlib.pyplot as plt


class LineFollowingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", "chart"], "render_fps": 4}

    def __init__(self, render_mode=None, sensor_number=5, motor_number=2):
        np.set_printoptions(suppress=True, precision=8)

        self.sensor_number = sensor_number
        self.motor_number = motor_number

        # straight along the center line (y-axis)
        self.current_angle = 0
        self._x_pos = 0
        self._y_pos = 0
        self._distance = 0
        self._distances = [self._distance]
        self._reward = 0
        self._rewards = [self._reward]

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
            self._center_location[int(self.sensor_number / 2)] = 0.5
            self._center_location[int((self.sensor_number / 2)) - 1] = 0.5
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

        # rendering human
        self.window = None
        self.clock = None
        # The size of the PyGame window
        self.window_size = 512

        # rendering chart
        self.fig = None
        self.ax = None
        self.line_distance = None
        self.line_reward = None
        # The size of Matplot window
        self.figsize = (10, 5)

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
        self._agent_location = self._center_location
        # self._agent_location = np.random.uniform(low=0.0, high=1.0, size=(self.sensor_number,))
        self._agent_location = self._convert_x_to_sensor_value(self._x_pos)
        self._target_location = self._center_location

        observation = self._get_obs()
        info = self._get_info()

        self._reward = np.exp(-info["distance"])
        self._rewards.append(self._reward)
        self._distance = info["distance"]
        self._distances.append(self._distance)

        self._render_frame()

        return observation, info

    def _simulate_next_location(self, left_speed, right_speed):
        # distance between wheels
        wheel_base = 1.0
        # forward speed
        v = (left_speed + right_speed) / 2.0
        # angular velocity
        # positive means left (counterclockwise), negative means right (clockwise)
        omega = (right_speed - left_speed) / wheel_base

        # calculate new position
        self._x_pos += v * np.sin(self.current_angle)
        self._y_pos += v * np.cos(self.current_angle)
        self.current_angle += omega

        vals = self._convert_x_to_sensor_value(self._x_pos)

        return vals

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

        if np.max(next_sensor_reading) == 0:
            return next_sensor_reading

        return next_sensor_reading / np.max(next_sensor_reading)

    def step(self, action):
        left_speed, right_speed = action

        self._agent_location = self._simulate_next_location(left_speed, right_speed)
        observation = self._get_obs()
        info = self._get_info()

        self._render_frame()

        terminated = np.array_equal(self._agent_location, self._target_location)

        self._reward = np.exp(-info["distance"])
        self._rewards.append(self._reward)
        self._distance = info["distance"]
        self._distances.append(self._distance)
        print(self._x_pos, info["distance"], self._reward, terminated, self._agent_location)

        return observation, self._reward, terminated, False, info

    def render(self):
        self._render_frame()

    def _before_render_human(self):
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))

        if self.clock is None:
            self.clock = pygame.time.Clock()

    def _after_render_human(self, canvas):
        self.window.blit(canvas, canvas.get_rect())
        pygame.event.pump()
        pygame.display.update()

        self.clock.tick(self.metadata["render_fps"])

    def _after_render_rgb_array(self, canvas):
        print(np.transpose(
            np.array(pygame.surfarray.pixels3d(canvas)),
            axes=(1, 0, 2)
        ))

    def _render_pygame(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # +2 to represent totally off the track position
        horizontal_square = (self.sensor_number + 2)
        pix_square_size = self.window_size / horizontal_square

        # draw center line
        x_center_line = ((horizontal_square / 2) * pix_square_size) - (pix_square_size / 2)
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(
                (x_center_line, 0),
                (pix_square_size, self.window_size),
            )
        )

        x_pos = (((horizontal_square / 2) + self._x_pos) * pix_square_size) - (
                (self.sensor_number / 2) * pix_square_size)
        pygame.draw.rect(
            canvas,
            (0, 0, 255),
            pygame.Rect(
                (x_pos, x_center_line),
                (self.sensor_number * pix_square_size, pix_square_size),
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

            return canvas

    def _before_render_matplot(self):
        if self.fig is None:
            plt.ion()

            self.fig, self.ax = plt.subplots(figsize=self.figsize)
            self.line_distance, = self.ax.plot([], [], label="Distance to Center Line", color='green', marker='o')
            self.line_reward, = self.ax.plot([], [], label="Reward", color='blue', marker='o')
            self.ax.legend()
            self.ax.set_xlim(0, 1000)
            self.ax.set_ylim(0, 10)

    def _render_matplot(self):
        print(self._distances, self._rewards)

        self.line_distance.set_data(list(range(0, len(self._distances))), self._distances)
        self.line_reward.set_data(list(range(0, len(self._rewards))), self._rewards)

        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)

    def _render_frame(self):
        if self.render_mode == "human":
            self._before_render_human()
        elif self.render_mode == "chart":
            self._before_render_matplot()

        if self.render_mode == "human" or self.render_mode == "rgb_array":
            canvas = self._render_pygame()

            if self.render_mode == "human":
                self._after_render_human(canvas)
            else:
                self._after_render_rgb_array(canvas)
        elif self.render_mode == "chart":
            self._render_matplot()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

        if self.fig is not None:
            plt.ioff()

# Improvement of the Gym environment with universe

import cv2
import numpy as np
import ale_py
import gymnasium as gym

from gymnasium.spaces import Box
from gymnasium.wrappers import RecordVideo
import os
from datetime import datetime


def create_atari_env(env_id, video=False,rank=None):
        
    # Generate a unique folder name for each run
    video_folder = None
    if video:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        video_folder = os.path.join("videos", f"run_{timestamp}")

    # Create the environment
    if video:
        env = gym.make(env_id, render_mode="rgb_array")
        # env = gym.make(env_id, render_mode="human")
    else:
        
        if rank == 0:
            env = gym.make(env_id, render_mode="human")
        else:
            env = gym.make(env_id, render_mode="rgb_array")
        

    # Wrap with video recording if requested
    if video:
        env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda ep: True)

    # Apply custom wrappers
    env = MyAtariRescale42x42(env)
    env = MyNormalizedEnv(env)

    return env


class MyAtariRescale42x42(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        # Update the observation space to reflect the rescaled frame dimensions
        self.observation_space = Box(low=0.0, high=1.0, shape=(1, 42, 42), dtype=np.float32)

    def observation(self, obs):
        # Process the observation using the helper function
        return self._process_frame42(obs)

    def _process_frame42(self, frame):
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized_frame = cv2.resize(gray_frame, (42, 42), interpolation=cv2.INTER_AREA)
        processed_frame = np.expand_dims(resized_frame, axis=0).astype(np.float32) / 255.0

        assert processed_frame.shape == (1, 42, 42), f"Shape incorrect: {processed_frame.shape}"
        return processed_frame



class MyNormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env=None):
        super().__init__(env)
        self.state_mean = 0
        self.state_std = 0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, obs):
        # Update step count
        self.num_steps += 1
        
        # Update running mean and standard deviation using exponential moving average
        self.state_mean = self.state_mean * self.alpha + obs.mean() * (1 - self.alpha)
        self.state_std = self.state_std * self.alpha + obs.std() * (1 - self.alpha)
        
        # Compute unbiased estimates
        unbiased_mean = self.state_mean / (1 - pow(self.alpha, self.num_steps))
        unbiased_std = self.state_std / (1 - pow(self.alpha, self.num_steps))
        
        # Normalize the observation
        normalized_obs = (obs - unbiased_mean) / (unbiased_std + 1e-8)
        
        # Expand dimensions if necessary (e.g., for grayscale images)
        return np.expand_dims(normalized_obs, axis=0)

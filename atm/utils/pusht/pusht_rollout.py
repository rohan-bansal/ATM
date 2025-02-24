from atm.utils.pusht.pusht_env import PushTEnv
from atm.dataloader.pusht_dataloader import PushTDataset
from typing import List
import numpy as np
import cv2
import os
import time
import pymunk
import pymunk.pygame_util

class PushTRolloutPipeline:
    def __init__(self, env: PushTEnv, dataloader: PushTDataset, video_path: str = None):
        self.env = env
        self.dataloader = dataloader
        self.reset_to = None

        self.video_path = video_path
        self.save_video = video_path is not None
        self.env.render_action = self.save_video

    # reset env to state [agent_x, agent_y, agent_z, block_x, block_y, block_z, block_angle]
    # will reuse this state unless changed or set to None
    def reset_to_state(self, state: np.ndarray = None):
        agent_xy = state[:2]
        block_xy = state[3:5]
        block_angle = [state[6]]
        print(agent_xy, block_xy, block_angle)
        reset_state = np.concatenate([agent_xy, block_xy, block_angle], axis=-1)
        self.reset_to = reset_state

    def _reset_env(self):
        self.env.reset_to_state = self.reset_to
        self.env.reset()

    def _save_video(self, video_frames: List[np.ndarray]):
        if self.video_path is None:
            return

        timestamp = int(time.time())
        video_path = os.path.join(self.video_path, f"rollout_{timestamp}.mp4")
        height, width = video_frames[0].shape[:2]
        video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

        for frame in video_frames:
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame_bgr)

        video_writer.release()
        

    # trajectory is [n x 3] where each row is [action_x, action_y, action_z]
    def rollout(self, trajectory: np.ndarray):

        video_frames = []

        self._reset_env()
        for i in range(trajectory.shape[0]):

            action_xy = trajectory[i, :2]
            action_xy_pygame = pymunk.pygame_util.to_pygame(action_xy, self.env.screen)
            obs, reward, done, info = self.env.step(action_xy_pygame)

            if self.save_video:
                img = self.env.render("rgb_array")
                video_frames.append(img)

            if done:
                break

        if self.save_video:
            self._save_video(video_frames)
        
        return obs, reward, done, info

    
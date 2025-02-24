import numpy as np

from atm.dataloader.track_dataloader import ATMPretrainDataset


class PushTDataset(ATMPretrainDataset):
    def __init__(self, sequence_length=16, *args, **kwargs):
        super().__init__(views=["agentview"], num_track_ts=16, num_track_ids=32, img_size=96, *args, **kwargs)
        self.sequence_length = sequence_length
        self.world_sized = 512
        self.render_size = 96


    def load_demo_info(self):
        start_idx = 0
        for demo_idx, fn in enumerate(self.buffer_fns):
            demo = self.load_h5(fn)

            demo_len = demo["root"]["agentview"]["raw"].shape[0]

            self._demo_id_to_path[demo_idx] = fn
            self._index_to_demo_id.update({k: demo_idx for k in range(start_idx, start_idx + demo_len*2)})
            self._index_to_view_id.update({k: (k - start_idx) % 2 for k in range(start_idx, start_idx + demo_len*2)})
            self._demo_id_to_start_indices[demo_idx] = start_idx
            self._demo_id_to_demo_length[demo_idx] = demo_len
            start_idx += demo_len * 2

        num_samples = len(self._index_to_demo_id)
        assert num_samples == start_idx

    def get_pixel_to_world(self, xy_pixel):
        # convert a pixel in the render image to the world coordinate
        # the render image is 96x96, and the world is 512x512
        # pixel (0, 0) maps to world (0, 0) [bottom left]
        # pixel (95, 95) maps to world (512, 512) [top right]
        
        if not (0 <= xy_pixel[0] < self.render_size and 0 <= xy_pixel[1] < self.render_size):
            raise ValueError(f"Pixel coordinates {xy_pixel} out of bounds. Must be within [0, {self.render_size-1}]")

        return np.array([xy_pixel[0] * self.world_sized / self.render_size,
                                xy_pixel[1] * self.world_sized / self.render_size])

    def get_agentview_images_at_index(self, index):
        demo_id = self._index_to_demo_id[index]
        images = self.get_agentview_images_per_demo(demo_id)

        # return images from index to sequence length. image is dict.
        return {k: v[index:index + self.sequence_length] for k, v in images.items()}

    def get_agentview_images_per_demo(self, demo_idx):

        demo_pth = self._demo_id_to_path[demo_idx]
        demo = self.load_h5(demo_pth)

        raw = demo["root"]["agentview"]["raw"]
        T_mask = demo["root"]["agentview"]["T"]
        background_mask = demo["root"]["agentview"]["background"]
        goal_mask = demo["root"]["agentview"]["goal"]
        pusher_mask = demo["root"]["agentview"]["pusher"]

        images = {
            "raw": raw,
            "T_mask": T_mask,
            "background_mask": background_mask,
            "goal_mask": goal_mask,
            "pusher_mask": pusher_mask,
        }

        return images
    
    def get_actions_at_index(self, index):
        demo_id = self._index_to_demo_id[index]
        actions = self.get_actions_at_demo(demo_id)
        return actions[index:index + self.sequence_length]
    
    def get_actions_at_demo(self, demo_idx):
        demo_pth = self._demo_id_to_path[demo_idx]
        demo = self.load_h5(demo_pth)

        actions = demo["root"]["actions"]
        
        return actions
    
    def get_state_at_index(self, index):
        demo_id = self._index_to_demo_id[index]
        agent_state, block_xyangle = self.get_state_at_demo(demo_id)
        return np.concatenate([agent_state[index:index + self.sequence_length], block_xyangle[index:index + self.sequence_length]], axis=-1)
    
    def get_state_at_demo(self, demo_idx):
        demo_pth = self._demo_id_to_path[demo_idx]
        demo = self.load_h5(demo_pth)

        agent_state = demo["root"]["extra_states"]["agent_xyz"]
        block_xyangle = demo["root"]["extra_states"]["block_xyz_angle"]
        
        return np.concatenate([agent_state, block_xyangle], axis=-1)


    def __getitem__(self, index):
        demo_id = self._index_to_demo_id[index]
        demo_start_index = self._demo_id_to_start_indices[demo_id]

        time_offset = (index - demo_start_index) // 2

        demo_pth = self._demo_id_to_path[demo_id]
        
        demo = self.load_h5(demo_pth)
        
        # Get the available length of the sequence
        available_len = len(demo["root"]["actions"]) - time_offset
        seq_len = min(self.sequence_length, available_len)
        
        # Extract and pad sequences 
        actions = np.pad(
            demo["root"]["actions"][time_offset:time_offset + seq_len],
            ((0, self.sequence_length - seq_len), (0, 0)),
            mode='edge'
        )
        agent_state = np.pad(
            demo["root"]["extra_states"]["agent_xyz"][time_offset:time_offset + seq_len],
            ((0, self.sequence_length - seq_len), (0, 0)),
            mode='edge'
        )
        block_xyangle = np.pad(
            demo["root"]["extra_states"]["block_xyz_angle"][time_offset:time_offset + seq_len],
            ((0, self.sequence_length - seq_len), (0, 0)),
            mode='edge'
        )

        return actions, agent_state, block_xyangle
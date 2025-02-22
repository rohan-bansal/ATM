import numpy as np
from diffusion_policy.dataset.pusht_image_dataset import PushTImageDataset
import cv2
import os
import h5py
from tqdm import tqdm


def apply_color_filtering(image: np.ndarray, tolerance: float = 0.05):

    pusher_color = np.array([64.0/255.0, 104.0/255.0, 224.0/255.0], dtype=np.float32)       # blue
    T_color = np.array([118.0/255.0, 134.0/255.0, 152.0/255.0], dtype=np.float32)        # gray
    goal_color = np.array([145.0/255.0, 239.0/255.0, 143.0/255.0], dtype=np.float32)           # green
    background_color = np.array([1, 1, 1], dtype=np.float32)     # white

    mask_pusher = np.linalg.norm(image - pusher_color, axis=-1) < tolerance
    mask_T = np.linalg.norm(image - T_color, axis=-1) < tolerance
    mask_goal = np.linalg.norm(image - goal_color, axis=-1) < tolerance

    mask_background = np.logical_and(
        np.linalg.norm(image - background_color, axis=-1) < 0.3,
        ~(mask_pusher | mask_T | mask_goal)
    )

    masks = {
         "pusher": mask_pusher,
         "T": mask_T,
         "goal": mask_goal,
         "background": mask_background,
    }
    
    return masks

def erode_masks(masks: dict, erosion_kernel: np.ndarray = None, iterations: int = 1):

    eroded_masks = {}
    if erosion_kernel is None:
        erosion_kernel = np.ones((2, 2), dtype=np.uint8)
    
    for key, mask in masks.items():
        # Convert boolean mask to uint8 (0 or 255)
        mask_uint8 = mask.astype(np.uint8) * 255
        # Apply erosion using OpenCV
        mask_eroded = cv2.erode(mask_uint8, erosion_kernel, iterations=iterations)
        # Convert back to a boolean mask
        eroded_masks[key] = mask_eroded > 0
    return eroded_masks

if __name__ == '__main__':
    zarr_path = os.path.expanduser('data/pusht/pusht_cchi_v7_replay.zarr')
    dataset = PushTImageDataset(zarr_path, horizon=16)

    base_dir = 'pusht'
    os.makedirs(base_dir, exist_ok=True)


    # Get episode lengths and ends
    episode_ends = dataset.replay_buffer.episode_ends[:]
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    n_episodes = len(episode_ends)

    # Convert each episode
    for ep_idx in tqdm(range(n_episodes)):
        start_idx = episode_starts[ep_idx]
        end_idx = episode_ends[ep_idx]
        ep_len = end_idx - start_idx

        # Create HDF5 file
        h5_path = os.path.join(base_dir, f'demo_{ep_idx}.hdf5')
        with h5py.File(h5_path, 'w') as f:
            # Create root group
            root = f.create_group('root')
            
            # Save actions under root
            actions = dataset.replay_buffer['action'][start_idx:end_idx]
            root.create_dataset('actions', data=actions)

            # Create agentview group under root
            agentview = root.create_group('agentview')

            img = dataset.replay_buffer['img'][start_idx:end_idx] / 255.0
            masks = apply_color_filtering(img, tolerance=0.2)
            eroded_masks = erode_masks(masks)

            # img = img.astype(np.uint8)
            # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            agentview.create_dataset('raw', data=img)
            for mask_name, mask in eroded_masks.items():
                agentview.create_dataset(mask_name, data=mask)

            
            # Create extra_states group under root
            extra_states = root.create_group('extra_states')
            states = dataset.replay_buffer['state'][start_idx:end_idx]
            
            # Split state into agent_xy and block_xy_angle
            extra_states.create_dataset('agent_xy', data=states[:,:2])
            extra_states.create_dataset('block_xy_angle', data=states[:,2:])
            # Save actions
            actions = dataset.replay_buffer['action'][start_idx:end_idx]
            f.create_dataset('actions', data=actions)

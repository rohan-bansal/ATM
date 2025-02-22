from atm.dataloader import PushTDataset
import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys
np.set_printoptions(threshold=sys.maxsize)

if __name__ == "__main__":
    dataset = PushTDataset(
        dataset_dir="/home/terra/dev/Research/rl2/atm/ATM/data/pusht",
    )
    # images = dataset.get_agentview_images_at_index(0)
    # image_1 = images["raw"][0]
    # # image_1 = image_1.astype(np.uint8)
    # plt.imshow(image_1)
    # plt.show()

    # mask_1 = images["T_mask"][0]
    # plt.imshow(mask_1)
    # plt.show()

    # image_1[~mask_1,:] = [0,0,0]

    # plt.imshow(image_1)
    # plt.show()

    # actions = dataset.get_actions_per_demo(0)
    # print(actions)

    pixel_to_world = dataset.get_pixel_to_world((24, 48))
    print(pixel_to_world)
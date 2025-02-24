from atm.dataloader import PushTDataset
from atm.utils.pusht.pusht_rollout import PushTRolloutPipeline
from atm.utils.pusht.pusht_env import PushTEnv

if __name__ == "__main__":
    dataset = PushTDataset(
        dataset_dir="/home/terra/dev/Research/rl2/atm/ATM/data/pusht",
    )
    
    env = PushTEnv()
    demo = 25;
    
    rollout_pipeline = PushTRolloutPipeline(env=env, dataloader=dataset, video_path="./tests")
    
    reset = dataset.get_state_at_demo(demo)
    rollout_pipeline.reset_to_state(reset[0])
    obs, reward, done, info = rollout_pipeline.rollout(dataset.get_actions_at_demo(demo))
    print(done, info)


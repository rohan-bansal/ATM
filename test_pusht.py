import robosuite
import numpy as np

# Create environment instance
env = robosuite.make(
    env_name="PushT",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)

# Reset the environment
obs = env.reset()

# Loop for some steps
for i in range(1000):
    action = np.zeros(env.robots[0].dof)  # Zero action to just view environment
    obs, reward, done, info = env.step(action)
    env.render()
    
    if done:
        obs = env.reset()

# Close the environment
env.close()

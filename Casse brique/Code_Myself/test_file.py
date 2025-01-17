import torch
import torch.nn.functional as F
from envs import create_atari_env
from model import ActorCritic
from torch.autograd import Variable
import time
from collections import deque
import matplotlib.pyplot as plt
import numpy as np

def test(rank, params, shared_model):
    torch.manual_seed(params.seed + rank)
    env = create_atari_env(params.env_name, video=True)
    env.reset(seed=params.seed + rank)
    model = ActorCritic(env.observation_space.shape[0], env.action_space)
    model.eval()
    state, _ = env.reset()
    state = torch.from_numpy(state)
    reward_sum = 0
    done = True
    start_time = time.time()
    actions = deque(maxlen=100)
    episode_length = 0
    time.sleep(60)

    while True:
        episode_length += 1
        if done:
            model.load_state_dict(shared_model.state_dict())
            with torch.no_grad():
                cx = torch.zeros(1, 256)
                hx = torch.zeros(1, 256)
        else:
            with torch.no_grad():
                cx = cx.data
                hx = hx.data

        value, action_value, (hx, cx) = model((state, (hx, cx)))
        
        prob = F.softmax(action_value, dim=1)

        
        # epsilon = 0.01 
        # if np.random.rand() < epsilon:
        #     action = np.random.choice(env.action_space.n)
        # else:
        action = prob.max(1)[1].data.numpy()

        # Execute the action in the environment
        state, reward, done, truncated, info = env.step(action[0])
        reward_sum += reward
        
        if reward != 0:
            episode_length = 0
            
        done = (done or episode_length >= params.max_episode_length)

        if done:
            result = "Time {}, episode reward {}, episode length {}\n".format(
                time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time)), reward_sum, episode_length
            )
            
            # Afficher dans la console (facultatif)
            print(result.strip())
            with open("result.txt", "a") as file:
                file.write(result)
            
            reward_sum = 0
            episode_length = 0
            actions.clear()
            state, _ = env.reset()
            time.sleep(60)

        state = torch.from_numpy(state)



        # Save or display the state image from time to time
        # if frame_count % 100 == 0:  # Every 100 frames
        #     # Check the state shape to understand its structure
        #     print(f"State shape: {state.shape}")

        #     # If state has 4 dimensions, we can remove the first dimension (batch dimension)
        #     if len(state.shape) == 4:
        #         state_image = state.squeeze(0)  # Remove batch dimension (1, C, H, W) -> (C, H, W)
        #     elif len(state.shape) == 3:
        #         state_image = state  # Correct shape (C, H, W)

        #     # Convert the state to an image (assuming state is a numpy array)
        #     state_image = state_image.permute(1, 2, 0).numpy()  # Rearrange if necessary, depending on the shape (C, H, W)
        #     plt.imshow(state_image)
        #     plt.title(f"Episode: {episode_length}, Frame: {frame_count}")
        #     plt.show()  # Display the image (can be commented out to avoid constant display)
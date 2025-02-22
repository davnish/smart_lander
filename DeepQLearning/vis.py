from DQ_train import createNetwork
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from moviepy.editor import VideoFileClip
import torch
import time
import os
import glob
import numpy as np


def Simulate_DQ_Strategy(model_no, games = 6):
    env = gym.make('LunarLander-v3', render_mode='human')
    onlineNetwork = createNetwork(8, 4)
    onlineNetwork.load_state_dict(torch.load(os.path.join('models', f"DQ_{model_no}.pt")))
    for _ in range(games):
        state = env.reset()[0]
        env.render()
        rewards = 0
        terminated = False
        timeStamp = 0
        while not terminated:
            action = torch.argmax(onlineNetwork(torch.tensor(state), )).item()
            state, reward, terminated, _, _ = env.step(action)
            rewards += reward
            timeStamp += 1
            time.sleep(0.05)
            if terminated or timeStamp>300:
                time.sleep(1)
                break
        print(f"Simulation Reward: {rewards}")

def saving_video(model_no, games = 1):
    env = gym.make('LunarLander-v3', render_mode="rgb_array_list")
    env.action_space.seed(42)
    onlineNetwork = createNetwork(4, 2)
    onlineNetwork.load_state_dict(torch.load(os.path.join('models', f"DQ_{model_no}.pt")))
    for game in range(games):
        rewards = 0
        state = env.reset()[0]
        terminated = False
        while not terminated:
            action = torch.argmax(onlineNetwork(torch.tensor(state))).item()
            state, reward, terminated, _, _ = env.step(action)
            rewards += reward
            if terminated:
                save_video(env.render(),"videos", name_prefix=f"DQ_{model_no}", 
                    episode_trigger=lambda x: np.ones(games, dtype=bool)[x],
                    fps=env.metadata["render_fps"],
                    episode_index=game)
                break
        print(f"Simulation Reward: {rewards}")

if __name__ == "__main__":
    # for model_no in range(1,6):
    #     saving_video(model_no, games = 1)
    # for _video in glob.glob(os.path.join('videos', '*.mp4')):
    #     video = VideoFileClip(_video)
    #     video.write_gif(os.path.join('results', f'{os.path.basename(_video).split('.')[0].split('-')[0]}.gif'),fps=10,program='imageio')
    # os.system('rm -rf videos')
    Simulate_DQ_Strategy(3, games = 3)
    pass
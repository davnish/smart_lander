from Q_train import returnIndexState
import gymnasium as gym
from gymnasium.utils.save_video import save_video
from moviepy.editor import VideoFileClip
import numpy as np
import time
import os
import glob


def Simulate_Q_Strategy(model_name, games = 1):
    env=gym.make('CartPole-v1', render_mode="human")
    obtainedRewards=[]
    Qmatrix = np.load(os.path.join('models', f'Qmatrix_{model_name}.npy'))
    for _ in range(games):
        timeStamp=0
        currState=env.reset()[0]
        env.render()
        rewardEpi = 0
        terminated = False
        while not terminated:
            action=np.argmax(Qmatrix[returnIndexState(currState)])
            currState, reward, terminated,_,_ =env.step(action)
            rewardEpi += reward 
            time.sleep(0.05)
            if terminated and timeStamp<500:
                time.sleep(1)
                break
            timeStamp += 1
        obtainedRewards.append(rewardEpi)
    env.close()
    return obtainedRewards

def saving_video(model_no, games = 2):
    env=gym.make('CartPole-v1', render_mode="rgb_array_list")
    Qmatrix = np.load(os.path.join('models', f'Qmatrix_{model_no}.npy'))
    for game in range(games):
        rewardEpi = 0
        currState=env.reset()[0]
        terminated = False
        while not terminated:
            action=np.argmax(Qmatrix[returnIndexState(currState)])
            currState, reward, terminated,_,_ =env.step(action)
            rewardEpi += reward 
            if terminated:
                save_video(env.render(),"videos", name_prefix=f"Q_{model_no}", 
                    episode_trigger=lambda x: np.ones(games, dtype=bool)[x],
                    fps=env.metadata["render_fps"],
                    episode_index=game)
                break
        print(f"Reward: {rewardEpi}")
    env.close()



if __name__ == "__main__":
    # for model_no in range(1,5):
    #     saving_video(model_no)
    # for _video in glob.glob(os.path.join('videos', '*.mp4')):
    #     video = VideoFileClip(_video)
    #     video.write_gif(os.path.join('results', f'{os.path.basename(_video).split('.')[0].split('-')[0]}.gif'),fps=10,program='imageio')
    # os.system('rm -rf videos')
    Simulate_Q_Strategy(1)
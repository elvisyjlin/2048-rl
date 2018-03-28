import argparse

import numpy as np
from gym2048 import Gym2048

def parse_args():
    parser = argparse.ArgumentParser(description='RL on 2048')
    parser.add_argument('--agent', help='agent type', choices=['random', 'pg', 'dqn'], default='random')
    parser.add_argument('--train', help='do training', action='store_true')
    parser.add_argument('--test', help='do testing', action='store_true')
    parser.add_argument('--model_path', help='path to model', default='model.h5')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    gym = Gym2048()
    env = gym.make()
    if args.agent == 'random':
        from agents.core import AgentRandom
        agent = AgentRandom(env)
    elif args.agent == 'pg':
        from agents.core import AgentPG
        agent = AgentPG(env, args.model_path)
    else:
        from agents.core import AgentDQN
        agent = AgentDQN(env, args.model_path)

    if args.train:
        agent.train()
    if args.test:
        if args.agent != 'random':
            agent.load(args.model_path)

        num_episode = 100
        rewards = []
        scores = []
        max_blocks = []

        for e in range(num_episode):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = agent.make_action(state)
                state, reward, done, info = env.step(action)
                # env.render()
                episode_reward += reward
                rewards.append(episode_reward)
            scores.append(env.game.score)
            max_blocks.append(env.game.grid.max())
            print('Game #{} \tReward: {:.2f}\tScore: {}\tLargest block: {}'.format(
                e, episode_reward, scores[-1], max_blocks[-1]))
        print('Ran {} episodes.'.format(num_episode))
        print('Avg reward: {}.'.format(np.mean(rewards)))
        print('Avg score: {}.'.format(np.mean(scores)))
        print('Largest block: {}.'.format(np.max(max_blocks)))

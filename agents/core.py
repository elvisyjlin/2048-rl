import numpy as np
import os
import random
from collections import deque

class Agent(object):
    def __init__(self, env, model_path=None):
        self.env = env
        self.model_path = model_path

    def make_action(self, observation, test=True):
        return NotImplementedError('This should be implemented!')

    def train(self):
        return NotImplementedError('This should be implemented!')

class AgentRandom(Agent):
    def __init__(self, env):
        super(AgentRandom, self).__init__(env)

    def make_action(self, observation, test=True):
        return self.env.action_space.sample()

    def train(self):
        pass

class AgentPG(Agent):
    def __init__(self, env, model_path):
        super(AgentPG, self).__init__(env, model_path)

        self.state_size = (16, )
        self.action_size = 4
        self.gamma = 0.9
        self.learning_rate = 1e-6
        self.save_interval = 100

        self.states = []
        self.gradients = []
        self.rewards = []
        self.probs = []

        from .networks import SimpleNetwork
        self.network = SimpleNetwork(self.state_size, self.action_size, 
            'softmax', 'categorical_crossentropy', self.learning_rate)

        if os.path.exists(self.model_path):
            print('loading saved model...')
            self.load(self.model_path)

    def remember(self, state, action, prob, reward):
        y = np.zeros([self.action_size])
        y[action] = 1
        self.gradients.append(y.astype(np.float32) - prob) 
        self.states.append(state)
        self.rewards.append(reward)

    def act(self, observation, test=True):
        observation = np.expand_dims(observation, axis=0)
        aprob = self.network.predict(observation, batch_size=1).flatten()
        if not test:
            self.probs.append(aprob)
        prob = aprob / np.sum(aprob)
        action = np.random.choice(self.action_size, 1, p=prob)[0]
        return action, prob

    def train_network(self):
        gradients = np.vstack(self.gradients)
        rewards = discount_rewards(np.vstack(self.rewards), self.gamma)
        rewards = normalize_rewards(rewards)
        gradients *= rewards
        # print(gradients.shape)

        X = np.vstack([self.states])
        y = self.probs + self.learning_rate * np.vstack([gradients])
        # print(len(self.probs), np.vstack([gradients]).shape)
        # print(X.shape)
        # print(y.shape)
        # import time
        # time.sleep(10)
        step_size = X.shape[0]
        loss = self.network.train_on_batch(X, y)
        self.states, self.probs, self.gradients, self.rewards = [], [], [], []
        return step_size, loss

    def train(self):
        state = self.env.reset()
        prev_x = None
        score = 0
        episode = 0
        step = 0

        while True:
            X = preprocess_observation(state)
            action, prob = self.act(X, False)
            state, reward, done, info = self.env.step(action)
            # print(action, prob, reward)
            self.env.render()
            score += reward
            self.remember(X, action, prob, reward)
            step += 1

            if done:
                episode += 1
                step_size, loss = self.train_network()
                print('Episode {} -\tReward: {},\tStep: {},\tLoss: {}'.format(
                    episode, score, step_size, loss))
                print('Game score: {}'.format(self.env.game.score))

                state = self.env.reset()
                prev_x = None
                score = 0
                step = 0

                if episode % self.save_interval == 0:
                    print('saving trained model...')
                    self.save(self.model_path)
            elif step > 100:
                step_size, loss = self.train_network()
                # print('Episode {} -\tReward: {},\tStep: {},\tLoss: {}'.format(
                #     episode, score, step_size, loss))
                step = 0

    def make_action(self, observation, test=True):
        action, _ = self.act(observation, test)
        return action
    
    def save(self, name):
        self.network.save_weights(name)
    
    def load(self, name):
        self.network.load_weights(name)

class AgentDQN(Agent):
    def __init__(self, env, model_path):
        super(AgentDQN, self).__init__(env, model_path)

        self.t = 0

        self.state_size = (16, )
        self.action_size = 4
        self.memory = deque()
        self.memory_size = 10000

        self.gamma = 0.99

        self.learning_rate = 1e-6
        self.epsilon_init = 1.0
        self.epsilon_min = 0.05
        self.exploration_steps = 1e+6
        self.epsilon_step = (self.epsilon_init - self.epsilon_min) / self.exploration_steps
        self.epsilon = self.epsilon_init

        self.initial_replay_size = 1e+4
        self.replay_interval = 4
        self.target_update_interval = 1000
        self.save_interval = 5e+4

        from .networks import SimpleNetwork
        self.main_network = SimpleNetwork(self.state_size, self.action_size, 
            'linear', 'mse', self.learning_rate)
        self.target_network = SimpleNetwork(self.state_size, self.action_size, 
            'linear', 'mse', self.learning_rate)

        if os.path.exists(self.model_path):
            print('loading saved model...')
            self.load(self.model_path)

        self.update_target_network()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if len(self.memory) > self.memory_size:
            self.memory.popleft()

    def act(self, observation, test=True):
        if test:
            observation = preprocess_observation(observation)
            observation = np.log2(observation)
            if np.random.rand() <= self.epsilon_min:
                action = random.randrange(self.action_size)
            else:
                action = np.argmax(self.main_network.predict(np.expand_dims(observation, axis=0), batch_size=1)[0])
            return action

        if np.random.rand() <= self.epsilon or not test and self.t < self.initial_replay_size:
            action = random.randrange(self.action_size)
        else:
            action = np.argmax(self.main_network.predict(np.expand_dims(observation, axis=0), batch_size=1)[0])

        if self.epsilon > self.epsilon_min and self.t >= self.initial_replay_size:
            self.epsilon -= self.epsilon_step

        return action

    def replay(self, batch_size):
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        y = []
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)
        states = np.stack(states)
        actions = np.stack(actions)
        rewards = np.stack(rewards)
        next_states = np.stack(next_states)
        dones = np.array(dones) + 0

        target_q_values = self.target_network.predict(next_states, batch_size=batch_size)
        max_target_q_values = np.amax(target_q_values, axis=1)
        max_target_q_values = rewards + (1 - dones) * self.gamma * max_target_q_values

        target_f = self.main_network.predict(states, batch_size=batch_size)
        target_f[range(batch_size), actions] = max_target_q_values

        hist = self.main_network.fit(states, target_f, batch_size=batch_size, verbose=0)
        loss = hist.history['loss'][0]
        return loss

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())

    def train(self):
        state = self.env.reset()
        total_reward = 0
        total_q_max = 0
        total_loss = 0
        duration = 0
        episode = 0

        while True:
            X = preprocess_observation(state)
            X[X!=0] = np.log2(X[X!=0])
            action = self.act(X, False)
            next_state, reward, done, info = self.env.step(action)
            next_X = preprocess_observation(next_state)
            next_X[next_X!=0] = np.log2(next_X[next_X!=0])
            if reward > 0: reward = np.log2(reward)
            self.remember(X, action, reward, next_X, done)
            state = next_state

            if self.t >= self.initial_replay_size:
                if self.t % self.replay_interval == 0:
                    loss = self.replay(32)
                    total_loss += loss
                if self.t % self.target_update_interval == 0:
                    self.update_target_network()
                if self.t % self.save_interval == 0:
                    print('saving trained model...')
                    self.save(self.model_path)

            total_reward += reward
            total_q_max += np.amax(self.main_network.predict(
                np.expand_dims(next_X, axis=0), 
                batch_size=1)[0])
            duration += 1

            if done:
                episode += 1

                if self.t < self.initial_replay_size:
                    mode = 'random'
                elif self.initial_replay_size <= self.t < self.initial_replay_size+self.exploration_steps:
                    mode = 'explore'
                else:
                    mode = 'exploit'

                print('Episode {} -\tTime step: {},\tDuration: {},\tEpsilon: {},\tTotal reward: {},\tAvg. Q max: {},\tAvg. loss: {},\tMode: {}'.format(
                    episode, self.t, duration, self.epsilon, total_reward, total_q_max/float(duration), 
                    total_loss/(float(duration)/self.replay_interval), mode))
                print('Game score: {}'.format(self.env.game.score))

                state = self.env.reset()
                total_reward = 0
                total_q_max = 0
                total_loss = 0
                duration = 0
            self.t += 1

    def make_action(self, observation, test=True):
        action = self.act(observation, test)
        return action
    
    def save(self, name):
        self.main_network.save_weights(name)
    
    def load(self, name):
        self.main_network.load_weights(name)

def preprocess_observation(observation):
    return observation.flatten()

def discount_rewards(rewards, gamma):
    discounted_rewards = np.zeros_like(rewards)
    running_add = 0
    for t in reversed(range(rewards.size)):
        if rewards[t] != 0:
            running_add = 0
        running_add = running_add * gamma + rewards[t]
        discounted_rewards[t] = running_add
    return discounted_rewards

def normalize_rewards(rewards):
    rewards_mean = np.mean(rewards)
    rewards_std = np.std(rewards)
    normalized_rewards = rewards - rewards_mean
    if rewards_std != 0:
        normalized_rewards = normalized_rewards / rewards_std
    return normalized_rewards
from collections import defaultdict
import random
from tqdm import tqdm
import numpy as np
import torch

# Abrstact agent class
class Agent:
    def __init__(self, env, agent_params):
        self.env = env

        self.alpha = agent_params.get("alpha", 0.1)
        self.gamma = agent_params.get("gamma", 0.9)
        self.epsilon = agent_params.get("epsilon", 0.1)
        self.epsilon_min = agent_params.get("epsilon_min", 0.01)
        self.epsilon_decay = agent_params.get("epsilon_decay", 1.0)
        self.num_actions = self.env.action_space.n
       
        self.q_values = defaultdict(lambda: np.zeros(self.num_actions))

        self.idx_states = {}

        self.rand_generator = np.random.RandomState(agent_params.get("seed", 42))

        self.reward_history = []
        self.score_history = []

    def update_epsilon(self):
        self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)

    def argmax(self, q_values):
        top = float("-inf")
        ties = []

        for i in range(len(q_values)):
            if q_values[i] > top:
                top = q_values[i]
                ties = [i]

            if q_values[i] == top:
                ties.append(i)

        return self.rand_generator.choice(ties)
    
    def getPolicy(self):
        policy = defaultdict(lambda: 0)
        for state in self.idx_states.keys():
            idx_state = self.idx_states[state]
            policy[state] = self.argmax(self.q_values[state])
        return policy
    
    def train(self, num_episodes, max_score):
        pass

# Abstract class for TD based agents
class OnlineAgent(Agent):

    def __init__(self, env, agent_params):
        super().__init__(env, agent_params)

    def agent_start(self, state):
        # Decays exploration as the episode number increases
        self.update_epsilon()

        if state not in self.idx_states.keys():
            self.idx_states[state] = len(self.idx_states)
        
        idx_state = self.idx_states[state]
        # Choose action using epsilon greedy.
        current_q = self.q_values[state]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions) # random action selection
        else:
            action = self.argmax(current_q) # greedy action selection
        self.prev_state = state
        self.prev_action = action
        return action

    def agent_step(self, reward, state):
        pass

    def agent_end(self, reward):
        pass

    def train(self, num_episodes, max_score):
        for i in tqdm(range(num_episodes)):
            state = self.env.reset()
            state = state[0]
            state = tuple(np.array(state).reshape(-1)) # Trick in order to make states hashable both for base and screen envs
            action = self.agent_start(state)
            total_reward = 0
            done = False
            while not done:
                next_state, reward, done, _, info = self.env.step(action)
                next_state = tuple(np.array(next_state).reshape(-1))
                total_reward += reward

                if info["score"] >= max_score:
                    self.agent_end(reward)
                    break
                state = next_state
                action = self.agent_step(reward, state)
            
            self.reward_history.append(total_reward)
            self.score_history.append(info["score"])

        return self.getPolicy(), self.score_history, self.reward_history
    

class OfflineAgent(Agent):

    def __init__(self, env, agent_params):
        super().__init__(env, agent_params)

    def get_probs(self, Q_s):
        """ obtains the action probabilities corresponding to epsilon-greedy policy """
        policy_s = np.ones(self.num_actions) * self.epsilon / self.num_actions
        best_a = np.argmax(Q_s)
        policy_s[best_a] = 1 - self.epsilon + self.epsilon/self.num_actions
        return policy_s

    def generate_episode_from_Q(self, env, max_score):
        """ generates an episode from following the epsilon-greedy policy """
        episode = []
        total_reward = 0
        state = env.reset()
        state = state[0]
        state = tuple(np.array(state).reshape(-1))
        if state not in self.idx_states.keys():
            self.idx_states[state] = len(self.idx_states)
        state_idx = self.idx_states[state]
        while True:
            action = np.random.choice(np.arange(self.num_actions), p=self.get_probs(self.q_values[state]))
            # take a step in the environement
            next_state, reward, done, _, info = env.step(action)
            next_state = tuple(np.array(next_state).reshape(-1))
            total_reward += reward
            episode.append((state, action, reward))
            state = next_state

            # Adds the state to the index
            if state not in self.idx_states.keys():
                self.idx_states[state] = len(self.idx_states)
            state_idx = self.idx_states[state]
            if done or info["score"] >= max_score:
                self.score_history.append(info["score"])
                self.reward_history.append(total_reward)
                break
        return episode
    
    def update_Q(self, episode):
        pass

    def train(self, num_episodes, max_score):
        for i_episode in tqdm(range(1, num_episodes+1)):
            # generate an episode by following epsilon-greedy policy
            episode = self.generate_episode_from_Q(self.env, max_score)

            # update the action-value function estimate using the episode
            self.update_Q(episode)
            self.update_epsilon()
        return self.getPolicy(), self.score_history, self.reward_history

    
    
class QLearningAgent(OnlineAgent):
    
    def __init__(self, env, agent_params):
        super().__init__(env, agent_params)
        
    
    def agent_step(self, reward, state):

        if state not in self.idx_states.keys():
            self.idx_states[state] = len(self.idx_states)
        
        idx_state = self.idx_states[state]
        
        # Choose action using epsilon greedy.
        current_q = self.q_values[state]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions) # random action selection
        else:
            action = self.argmax(current_q)

        # Update Q-values according to the greedy policy
        self.q_values[self.prev_state][self.prev_action] += self.alpha * (reward + self.gamma * np.max(current_q) - self.q_values[self.prev_state][self.prev_action])

        self.prev_state = state
        self.prev_action = action

        return action
    
    def agent_end(self, reward):
        # Update Q-values
        self.q_values[self.prev_state][self.prev_action] += self.alpha * (reward - self.q_values[self.prev_state][self.prev_action])

class ExpectedSarsaAgent(OnlineAgent):

    def __init__(self, env, agent_params):
        super().__init__(env, agent_params)
        
    
    def agent_step(self, reward, state):

        if state not in self.idx_states.keys():
            self.idx_states[state] = len(self.idx_states)
        
        idx_state = self.idx_states[state]
        
        # Choose action using epsilon greedy.
        current_q = self.q_values[state]
        if self.rand_generator.rand() < self.epsilon:
            action = self.rand_generator.randint(self.num_actions) # random action selection
        else:
            action = self.argmax(current_q)

        # Computes the expected Q-value of the current state
        expected = 0
        for a in range(self.num_actions):
            if a == np.argmax(current_q):
              expected += (1-self.epsilon + self.epsilon/self.num_actions)*current_q[a]
            else:
              expected += (self.epsilon/self.num_actions)*current_q[a]

        # Update Q-values according to the greedy policy
        self.q_values[self.prev_state][self.prev_action] += self.alpha * (reward + self.gamma * expected - self.q_values[self.prev_state][self.prev_action])

        self.prev_state = state
        self.prev_action = action

        return action
    
    def agent_end(self, reward):
        # Update Q-values
        self.q_values[self.prev_state][self.prev_action] += self.alpha * (reward - self.q_values[self.prev_state][self.prev_action])

        return
    
class SarsaLambdaAgent(OnlineAgent):
    
        def __init__(self, env, agent_params):
            super().__init__(env, agent_params)
            
            self.lambda_value = agent_params.get("lambd", 0.9)
            self.eligibility_traces = defaultdict(lambda: np.zeros(self.num_actions))

        def agent_step(self, reward, state):

            if state not in self.idx_states.keys():
                self.idx_states[state] = len(self.idx_states)
            
            idx_state = self.idx_states[state]
            
            # Choose action using epsilon greedy.
            current_q = self.q_values[state]
            # print(current_q.shape)
            if self.rand_generator.rand() < self.epsilon:
                action = self.rand_generator.randint(self.num_actions) # random action selection
            else:
                action = self.argmax(current_q)

            delta = reward + self.gamma*current_q[action] - self.q_values[self.prev_state][self.prev_action]
    
            self.eligibility_traces[self.prev_state][self.prev_action] += 1

            for s in self.q_values.keys():
                self.q_values[s] += self.alpha * delta * self.eligibility_traces[s] # Update Q
                self.eligibility_traces[s] *= self.gamma * self.lambda_value # Update eligibility traces
        
            self.prev_state = state
            self.prev_action = action

            return action

        def agent_end(self, reward):
            # Update Q-values
            delta = reward - self.q_values[self.prev_state][self.prev_action]
            for s in self.q_values.keys():
                self.q_values[s] += self.alpha * delta * self.eligibility_traces[s]
                self.eligibility_traces[s] = np.zeros_like(self.eligibility_traces[s])
            return
        
class MCControlAgent(OfflineAgent):
    
    def __init__(self, env, agent_params):
        super().__init__(env, agent_params)
    
    def update_Q(self, episode):
        """ updates the action-value function estimate using the most recent episode """
        states, actions, rewards = zip(*episode)
        # prepare for discounting
        discounts = np.array([self.gamma**i for i in range(len(rewards)+1)])
        for i, state in enumerate(states):
            state_idx = self.idx_states[state]
            old_Q = self.q_values[state][actions[i]]
            self.q_values[state][actions[i]] = old_Q + self.alpha*(sum(rewards[i:]*discounts[:-(i+1)]) - old_Q)

        return
    
# class DeepQLearningAgent(Agent):
#     def __init__(self, env, agent_params):
#         super(DeepQLearningAgent, self).__init__(env, agent_params)
        
#         self.epsilon = agent_params.get("epsilon", 0.1)
#         self.epsilon_min = agent_params.get("epsilon_min", 0.01)
#         self.epsilon_decay = agent_params.get("epsilon_decay", 0.995)
#         self.gamma = agent_params.get("gamma", 0.99)
#         self.alpha = agent_params.get("alpha", 0.01)
#         self.batch_size = agent_params.get("batch_size", 32)
#         self.memory = []
#         self.memory_capacity = agent_params.get("memory_capacity", 10000)
#         self.num_actions = env.action_space.n
#         self.dim_states = agent_params.get("dim_state", 2)

#         self.lr = 0.01
#         self.model = self.build_model()
#         self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
#         self.loss_fn = torch.nn.MSELoss()
#         self.rand_generator = np.random.RandomState(agent_params.get("seed", 42))

#     def build_model(self):
#         model = torch.nn.Sequential(
#             torch.nn.Linear(self.dim_states, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, 128),
#             torch.nn.ReLU(),
#             torch.nn.Linear(128, self.num_actions)
#         )
#         return model
    
#     def get_action(self, state):
#         if self.rand_generator.rand() < self.epsilon:
#             return self.rand_generator.randint(self.num_actions)
#         state = torch.tensor(state, dtype=torch.float).view(-1)
#         with torch.no_grad():
#             return torch.argmax(self.model(state)).item()
    
#     def remember(self, state, action, reward, next_state, done):
#         self.memory.append([state, action, reward, next_state, done])
#         if len(self.memory) > self.memory_capacity:
#             self.memory.pop(0)
    
#     def replay(self):
#         if len(self.memory) < self.batch_size:
#             return
#         batch = self.rand_generator.choice(len(self.memory), self.batch_size, replace=False)
#         for i in batch:
#             state, action, reward, next_state, done = self.memory[i]
#             state = torch.tensor(state, dtype=torch.float)
#             next_state = torch.tensor(next_state, dtype=torch.float)
#             target = reward
#             if not done:
#                 with torch.no_grad():
#                     target += self.gamma*torch.max(self.model(next_state.view(-1))).item()
#             target_f = self.model(state.view(-1))
#             target_f[action] = target
#             self.optimizer.zero_grad()
#             loss = self.loss_fn(self.model(state.view(-1)), target_f)
#             loss.backward()
#             self.optimizer.step()

#     def train(self, num_episodes, max_score):
#         for i in tqdm(range(num_episodes)):
#             state = self.env.reset()
#             state = state[0]
#             done = False
#             total_reward = 0
#             while not done:
#                 action = self.get_action(state)
#                 next_state, reward, done, _, info = self.env.step(action)
#                 self.remember(state, action, reward, next_state, done)
#                 state = next_state
#                 total_reward += reward
#             self.replay()
#             self.reward_history.append(total_reward)
#             self.score_history.append(info["score"])
#             self.epsilon = max(self.epsilon*self.epsilon_decay, self.epsilon_min)
#         return None, self.score_history, self.reward_history
        
        
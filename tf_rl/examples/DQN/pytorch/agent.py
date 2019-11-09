import torch
import os
import copy
import numpy as np
from tf_rl.examples.DQN.pytorch.network import net


class dqn_agent(object):
    def __init__(self, num_action, policy, summary_writer, learning_rate, gamma, model_path, cuda):
        # define some important
        self.timestep = 0
        self.eval_flg = False
        self.num_action = num_action
        self.cuda = cuda
        self.learning_rate = learning_rate
        self.gamma = gamma

        self.policy = policy
        self.summary_writer = summary_writer
        self.net = net(self.num_action)
        self.target_net = copy.deepcopy(self.net)
        self.target_net.load_state_dict(self.net.state_dict())
        if self.cuda:
            self.net.cuda()
            self.target_net.cuda()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=learning_rate)

        self.model_path = model_path
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

    def select_action(self, state):
        with torch.no_grad():
            state = self._get_tensors(state)
            action = self.policy.select_action(q_value_fn=self.net, state=state, ts=self.timestep)
        return action

    def select_action_eval(self, state, epsilon):
        with torch.no_grad():
            state = self._get_tensors(state)
            action = self.policy.select_action(q_value_fn=self.net, state=state, epsilon=epsilon, ts=self.timestep)
        return action

    def update(self, states, actions, rewards, next_states, dones):
        # convert the data to tensor
        states = self._get_tensors(states)
        actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(-1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(-1)
        next_states = self._get_tensors(next_states)
        dones = torch.tensor(1 - dones, dtype=torch.float32).unsqueeze(-1)
        # convert into gpu
        if self.cuda:
            actions = actions.cuda()
            rewards = rewards.cuda()
            dones = dones.cuda()
        loss = self._update(states, actions, rewards, next_states, dones)
        return loss

    def _update(self, states, actions, rewards, next_states, dones):
        # calculate the target value
        with torch.no_grad():
            target_action_value = self.target_net(next_states)
            target_action_max_value, _ = torch.max(target_action_value, dim=1, keepdim=True)
        # target
        expected_value = rewards + self.gamma * target_action_max_value * dones
        # get the real q value
        action_value = self.net(states)
        real_value = action_value.gather(1, actions)
        loss = (expected_value - real_value).pow(2).mean()
        # start to update
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # visualisation
        self.summary_writer.add_scalar('agent/mean_td_target', expected_value.mean(), self.timestep)
        self.summary_writer.add_scalar('agent/mean_q_t', action_value.mean(), self.timestep)
        self.summary_writer.add_scalar('agent/mean_q_tp1', target_action_value.mean(), self.timestep)
        self.summary_writer.add_scalar('agent/loss_td_error', loss.item(), self.timestep)
        self.summary_writer.add_scalar('agent/mean_diff_q_tp1_q_t', (target_action_value - action_value).mean(),
                                       self.timestep)
        self.summary_writer.add_scalar('train/Eps', self.policy.current_epsilon(self.timestep), self.timestep)

        return loss.item()

    # get tensors
    def _get_tensors(self, obs):
        if obs.ndim == 3:
            obs = np.transpose(obs, (2, 0, 1))
            obs = np.expand_dims(obs, 0)
        elif obs.ndim == 4:
            obs = np.transpose(obs, (0, 3, 1, 2))
        obs = torch.tensor(obs, dtype=torch.float32)
        if self.cuda:
            obs = obs.cuda()
        return obs

    def save(self):
        torch.save(self.net.state_dict(), self.model_path + '/model.pt')

    def sync_network(self):
        self.target_net.load_state_dict(self.net.state_dict())

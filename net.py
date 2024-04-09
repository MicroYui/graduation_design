import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter

import main


class Actor(nn.Module):
    def __init__(self, state_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(2, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, state_dim)

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.sigmoid(self.l3(a))
        return a


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state):
        q = F.relu(self.l1(state))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class TD3(object):
    def __init__(self):
        self.actor = Actor(28)
        # self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adagrad(self.actor.parameters(), lr=0.00001)

        self.critic = Critic(28)
        # self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.001)

        self.epoch_num = 100

    def train(self):
        writer = SummaryWriter()
        flat = torch.tensor([0.2, 0.8])
        for i in trange(100):
            state = self.actor(flat)
            state = state + torch.randn(28) * 0.01
            state = state.clamp(min=0, max=1)
            q = self.critic(state)
            actor_loss = q
            reward = main.get_reward(state)

            if i % 10 == 0:
                writer.add_scalar('reward', reward, i)
                writer.add_scalar('actorLoss', actor_loss, i)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            for _ in range(100):
                state = self.actor(flat)
                state = state + torch.randn(28) * 0.05
                state = state.clamp(min=0, max=1)
                reward = main.get_reward(state)
                critic_loss = (self.critic(state) - reward) ** 2

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                self.critic_optimizer.step()

        # torch.save(self.actor, 'actor.pkl')
        # torch.save(self.critic, 'critic.pkl')
        writer.close()

            # soft_update(self.actor_target, self.actor, 0.001)
            # soft_update(self.critic_target, self.critic, 0.1)


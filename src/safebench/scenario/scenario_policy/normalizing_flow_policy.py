''' 
Date: 2023-01-31 22:23:17
LastEditTime: 2023-03-22 17:26:29
Description: 
    Copyright (c) 2022-2023 Safebench Team

    This work is licensed under the terms of the MIT license.
    For a copy, see <https://opensource.org/licenses/MIT>

    This file implements the method proposed in paper:
        Multimodal Safety-Critical Scenarios Generation for Decision-Making Algorithms Evaluation
        <https://arxiv.org/pdf/2009.08311.pdf>
'''

import os
import numpy as np
from fnmatch import fnmatch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions.multivariate_normal import MultivariateNormal

from safebench.scenario.scenario_policy.reinforce_continuous import REINFORCE
from safebench.util.torch_util import CUDA, CPU


class MLP(nn.Module):
    def __init__(self, n_input, n_output, n_hidden):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3_s = nn.Linear(n_hidden, n_output)
        self.fc3_t = nn.Linear(n_hidden, n_output)
        
    def forward(self, x):
        hidden = F.relu(self.fc2(F.relu(self.fc1(x))))
        s = torch.tanh(self.fc3_s(hidden))
        t = self.fc3_t(hidden)
        return s, t


class ConditionalRealNVP(nn.Module):
    # Generator, condition_dim=self.state_dim, data_dim=self.action_dim
    def __init__(self, n_flows, condition_dim, data_dim, n_hidden):
        super(ConditionalRealNVP, self).__init__()
        self.n_flows = n_flows
        self.condition_dim = condition_dim

        # divide the data dimension by 1/2 to do the affine operation
        assert(data_dim % 2 == 0)
        self.n_half = int(data_dim/2)

        # build the network list
        self.NN = torch.nn.ModuleList()
        for k in range(n_flows):
            # the input of each layer should also contain the condition
            # self.n_half+self.condition_dim作为输入维度。
            # 这是因为在每个流中，在执行仿射变换之前，条件维度与数据的一半维度进行拼接（concatenation）。
            # 因此，该神经网络除了要处理原始数据的一半，还要处理条件数据。
            self.NN.append(MLP(self.n_half+self.condition_dim, self.n_half, n_hidden)) # 这个self.NN不会和后面的RealNVP冲突吗？？
        
    def forward(self, x, c):
        log_det_jacobian = 0
        for k in range(self.n_flows):
            x_a = x[:, :self.n_half]
            x_b = x[:, self.n_half:]
            
            x_a_c = torch.cat([x_a, c], dim=1)
            s, t = self.NN[k](x_a_c)
            x_b = torch.exp(s)*x_b + t
            
            x = torch.cat([x_b, x_a], dim=1)
            log_det_jacobian += s
        
        return x, log_det_jacobian
        
    def inverse(self, z, c):
        # 反向操作是将变换后的数据（这里表示为z,(mean)）重新映射回原始数据空间的流程。在这个上下文中，z表示已经转换过的数据，而c代表了条件变量。
        for k in reversed(range(self.n_flows)):
            z_a = z[:, self.n_half:]
            z_b = z[:, :self.n_half]

            z_a_c = torch.cat([z_a, c], dim=1)

            s, t = self.NN[k](z_a_c) # 使用第k个神经网络获取尺度参数s和平移参数t, self.NN[k]调用了第k个子模型，传入数据z_a_c进行前向计算，分别输出尺度和平移参数。
            z_b = (z_b - t) / torch.exp(s) # 使用尺度和平移参数更新z_b。在RealNVP的反向过程中，需要对z_b执行反运算，即减去平移参数t，然后除以exp(s)（因为在正向中是乘以exp(s)）。
            z = torch.cat([z_a, z_b], dim=1)
        return z


# for prior model，flow-based）模型，用于生成模型任务，如样本生成或概率密度估计，从简单分布（如高斯分布）有效地生成复杂数据分布的样本。
class RealNVP(nn.Module):
    def __init__(self, n_flows, data_dim, n_hidden): 
        # n_flows：流的数量。在RealNVP模型中，数据会依次通过这些流进行转换，每个流都尝试捕捉并建模数据的不同特征和依赖。
        super(RealNVP, self).__init__()
        self.n_flows = n_flows

        # divide the data dimension by 1/2 to do the affine operation
        # data_dim：数据的维度。它是每个数据点的特征或维度数量。==self.action_dim，NPC的动作维度？
        # n_hidden：隐藏层的维度。这在构建每个流使用的神经网络时使用。
        assert(data_dim % 2 == 0) # 一半的数据会用于条件变换，另一半进行仿射变换（affine transformation）
        self.n_half = int(data_dim/2) 

        # build the network list
        self.NN = torch.nn.ModuleList() # 在RealNVP模型中，每个“流（flow）”都是通过具体的神经网络来实现的，这里将用它来存储所有流的网络 
        for k in range(n_flows):
            # the input of each layer should also contain the condition
            # 对于每个流，通过循环创建一个MLP（多层感知器）网络，并将其添加到NN列表中。
            # 这个MLP的输入和输出维度都是self.n_half，表示每个流网络仅处理数据的一半。
            # n_hidden是传递给MLP的另一个参数，表示隐藏层的大小。
            self.NN.append(MLP(self.n_half, self.n_half, n_hidden))
        
    def forward(self, x):
        log_det_jacobian = 0
        for k in range(self.n_flows):
            x_a = x[:, :self.n_half]
            x_b = x[:, self.n_half:]

            s, t = self.NN[k](x_a)
            x_b = torch.exp(s)*x_b + t
            x = torch.cat([x_b, x_a], dim=1)
            log_det_jacobian += s
        
        return x, log_det_jacobian
        
    def inverse(self, z):
        for k in reversed(range(self.n_flows)):
            z_a = z[:, self.n_half:]
            z_b = z[:, :self.n_half]
            s, t = self.NN[k](z_a)
            z_b = (z_b - t) / torch.exp(s)
            z = torch.cat([z_a, z_b], dim=1)
        return z


class NormalizingFlow(REINFORCE):
    name = 'nf'
    type = 'init_state'

    def __init__(self, scenario_config, logger):
        self.logger = logger
        self.num_waypoint = 31
        self.continue_episode = 0
        self.num_scenario = scenario_config['num_scenario']
        self.model_path = os.path.join(scenario_config['ROOT_DIR'], scenario_config['model_path'])
        self.model_id = scenario_config['model_id']
        self.use_prior = scenario_config['use_prior'] # 使用先验与否

        self.lr = scenario_config['lr']
        self.batch_size = scenario_config['batch_size']
        self.prior_lr = scenario_config['prior_lr']

        self.prior_epochs = scenario_config['prior_epochs']
        self.alpha = scenario_config['alpha']
        self.itr_per_train = scenario_config['itr_per_train']

        self.state_dim = scenario_config['state_dim'] # 路点数量？63
        self.action_dim = scenario_config['action_dim'] # ？
        self.reward_dim = scenario_config['reward_dim'] # ？
        self.drop_threshold = scenario_config['drop_threshold']
        self.n_flows = scenario_config['n_flows_model'] # ？

        # latent space 创建多元正态（或高斯）分布的类。这个分布有两个参数——均值向量和协方差矩阵，均值向量，协方差矩阵及其精度矩阵（逆）和Cholesky分解（下三角）
        self.z = MultivariateNormal(CUDA(torch.zeros(self.action_dim)), CUDA(torch.eye(self.action_dim))) # --p(z|x), simple distribution, 

        # prior model and generator
        self.prior_model = CUDA(RealNVP(n_flows=self.n_flows, data_dim=self.action_dim, n_hidden=128))
        self.model = CUDA(ConditionalRealNVP(n_flows=self.n_flows, condition_dim=self.state_dim, data_dim=self.action_dim, n_hidden=64))
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def train_prior_model(self, prior_data):
        """ 
            Train the prior model using the data from the prior distribution.
            This function should be used seperately from the Safebench framework to train the prior model.
        """
        prior_data = CUDA(torch.tensor(prior_data))
        # papre a data loader
        train_loader = torch.utils.data.DataLoader(prior_data, shuffle=True, batch_size=self.batch_size)
        self.prior_optimizer = optim.Adam(self.prior_model.parameters(), lr=self.prior_lr)
        self.prior_model.train()

        # train the model
        for epoch in range(self.prior_epochs):
            avg_loglikelihood = []
            for data in train_loader:
                sample_z, log_det_jacobian = self.prior_model(data)
                log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdims=True)
                loglikelihood = -torch.mean(self.z.log_prob(sample_z)[:, None] + log_det_jacobian)
                self.prior_optimizer.zero_grad()
                loglikelihood.backward()
                self.prior_optimizer.step()
                avg_loglikelihood.append(loglikelihood.item())
            self.logger.log('[{}/{}] Prior training error: {}'.format(epoch, self.prior_epochs, np.mean(avg_loglikelihood)))

    def prior_likelihood(self, actions):
        sample_z, log_det_jacobian = self.prior_model(actions)
        log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdims=True)
        loglikelihood = self.z.log_prob(sample_z)[:, None] + log_det_jacobian
        prob = torch.exp(loglikelihood)
        return prob

    def flow_likelihood(self, actions, condition):
        sample_z, log_det_jacobian = self.model(actions, condition)
        # make sure the dimension is aligned, for action_dim > 2, the log_det is more than 1 dimension
        log_det_jacobian = torch.sum(log_det_jacobian, dim=1, keepdims=True) # 雅可比矩阵
        loglikelihood = self.z.log_prob(sample_z)[:, None] + log_det_jacobian
        return loglikelihood

    def prior_sample(self, sample_number=1000, sigma=1.0): # 没有用到？
        sampler = MultivariateNormal(CUDA(torch.zeros(self.action_dim)), CUDA(sigma*torch.eye(self.action_dim)))
        new_sampled_z = sampler.sample((sample_number,))

        self.prior_model.eval()
        with torch.no_grad():
            prior_flow = self.prior_model.inverse(new_sampled_z)
        return prior_flow.cpu().numpy()

    def flow_sample(self, state, sample_number=1000, sigma=1.0): # 没有用到？
        # use a new sampler, then we can control the sigma 
        sampler = MultivariateNormal(CUDA(torch.zeros(self.action_dim)), CUDA(sigma*torch.eye(self.action_dim)))
        new_sampled_z = sampler.sample((sample_number,))

        # condition should be repeated sample_number times
        condition = CUDA(torch.tensor(state))
        condition = condition.repeat(sample_number, 1)

        self.model.eval()
        with torch.no_grad():
            action_flow = self.model.inverse(new_sampled_z, condition)
        return action_flow

    def get_init_action(self, state, deterministic=False):
        # the state should be a sequence of route waypoints + target_speed (31*2+1)
        processed_state = self.proceess_init_state(state)
        processed_state = CUDA(torch.from_numpy(processed_state))

        self.model.eval() # 确保模型在评估模式下运行，关闭Dropout和BatchNorm等影响模型表现的因素。
        with torch.no_grad(): # 将PyTorch的梯度计算暂时关闭，用于阻止Autograd（自动梯度）引擎从计算和存储梯度，从而提高内存效率并加速计算。
            # mean = CUDA(torch.zeros(self.action_dim))[None] # 创建一个由零组成的向量，形状与动作维度匹配，并将其转移到GPU上。同时，[None]用于增加一个额外的维度以匹配模型的输入需要。
            # condition = CUDA(torch.tensor(processed_state)) # 作为条件供模型生成动作,将路点作为condition?
            # action = self.model.inverse(mean, condition) # model 是 generator , mean->z,
            action_flow = self.flow_sample(processed_state,sigma=0.8)
            random_index = torch.randint(0, action_flow.size(0), (1,)).item()  # item() 获得Python数值
            action = action_flow[random_index][None]

        action_list = []
        for a_i in range(self.action_dim):
            action_list.append(action.cpu().numpy()[0, a_i])
        return [action_list], None

    # train on batched data
    def train(self, replay_buffer):
        if replay_buffer.init_buffer_len < self.batch_size:
            return

        self.model.train()
        # the buffer can be resued since we evaluate action-state every time 
        for _ in range(self.itr_per_train):
            # get episode reward
            batch = replay_buffer.sample_init(self.batch_size)
            state = batch['static_obs']
            action = batch['init_action']
            episode_reward = batch['episode_reward']
            
            processed_state = self.proceess_init_state(state) # caixuan 62 WP + 1 TS
            processed_state = CUDA(torch.from_numpy(processed_state))
            action = CUDA(torch.from_numpy(action))
            episode_reward = CUDA(torch.from_numpy(episode_reward))[None].t()

            loglikelihood = self.flow_likelihood(action, processed_state) # log P_x(x|\theta)
            prior_prob = self.prior_likelihood(action) if self.use_prior else 0 # P_x(x|\theta)
            assert loglikelihood.shape == episode_reward.shape

            # this term is actually the log-likelihood weighted by reward
            loss_r = -(loglikelihood * (torch.exp(-episode_reward) + self.alpha * prior_prob)).mean()
            self.optimizer.zero_grad()
            loss_r.backward()
            self.optimizer.step()

    def save_model(self,epoch):
        if not os.path.exists(self.model_path):
            self.logger.log(f'>> Creating folder for saving model: {self.model_path}')
            os.makedirs(self.model_path)
        model_filename = os.path.join(self.model_path, f'{self.model_id}.{epoch:04}.pt')
        self.logger.log(f'>> Saving nf model to {model_filename}')
        with open(model_filename, 'wb+') as f:
            torch.save({'parameters': self.model.state_dict()}, f)

    def load_model(self, episode=None):
        if episode is None:
            episode = -1
            for _, _, files in os.walk(self.model_path):
                for name in files:
                    if fnmatch(name, "*torch"):
                        cur_episode = int(name.split(".")[-2])
                        if cur_episode > episode:
                            episode = cur_episode
        model_filename = os.path.join(self.model_path, f'{self.model_id}.{episode:04}.pt')
        if os.path.exists(model_filename):
            self.logger.log(f'>> Loading nf model from {model_filename}')
            with open(model_filename, 'rb') as f:
                checkpoint = torch.load(f)
            self.model.load_state_dict(checkpoint['parameters'])
            self.continue_episode = episode
        else:
            self.logger.log(f'>> Fail to find nf model from {model_filename}', color='yellow')

    def save_prior_model(self):
        states = {'parameters': self.prior_model.state_dict()}
        model_filename = os.path.join(self.model_path, 'nf.prior.'+str(self.model_id)+'.pt')
        with open(model_filename, 'wb+') as f:
            torch.save(states, f)
            self.logger.log(f'>> Save prior model of nf')

    def load_prior_model(self):
        model_filename = os.path.join(self.model_path, 'nf.prior.'+str(self.model_id)+'.pt')
        self.logger.log(f'>> Loading nf model from {model_filename}')
        if os.path.isfile(model_filename):
            with open(model_filename, 'rb') as f:
                checkpoint = torch.load(f)
            self.prior_model.load_state_dict(checkpoint['parameters'])
        else:
            self.logger.log(f'>> Fail to find nf prior model from {model_filename}', color='yellow')

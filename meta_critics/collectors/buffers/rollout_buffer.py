import torch


class RolloutBuffer:
    def __init__(self, num_envs, num_steps, single_observation_dim, single_action_dim, device="cpu"):
        """

        :param num_envs:
        :param num_steps:
        :param device:
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.single_observation_dim = single_observation_dim
        self.single_action_dim = single_action_dim
        self.device = device

        #
        self.obs = torch.zeros((num_steps, num_envs) + single_observation_dim).to(device)
        self.actions = torch.zeros((num_steps, num_envs) + single_action_dim).to(device)
        self.logp = torch.zeros((num_steps, num_envs)).to(device)
        self.rewards = torch.zeros((num_steps, num_envs)).to(device)
        self.dones = torch.zeros((num_steps, num_envs)).to(device)
        self.values = torch.zeros((num_steps, num_envs)).to(device)

    def clean(self):
        self.obs = torch.zeros((self.num_steps, self.num_envs) + self.single_observation_dim).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs) + self.single_action_dim).to(self.device)
        self.logp = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

    def print_shapes(self):
        print("Observation \t", self.obs.shape)
        print("Action space \t", self.actions.shape)
        print("Rewards \t", self.actions.shape)
        print("Dones\t", self.actions.shape)
        print("Values\t", self.values.shape)
        print("single observation\t", self.actions.shape)
        print("single action\t", self.actions.shape)

    def print_action(self):
        print("Action space \t", self.actions)

    def print_observation(self):
        print("Observation \t", self.obs)

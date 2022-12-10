# https://arxiv.org/abs/1807.09809
#
# Section 4 Experiment
# In the below experiments for the contextual models (including
# the epsilon-greedy and fixed dropout rate bandits),
# we use a neural network with 2 hidden layers with
# 256 units each and ReLU activation function. All networks
# are trained using the Adam optimizer

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from meta_critics.agents.bandits.modules.memory import Memory


class BayesianLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device="cpu") -> None:
        """
        :param in_features: size of each input sample
        :param out_features: size of each output sample
        :param bias: Whether to use an additive bias. Defaults to True.
        """
        super(BayesianLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.w_mu = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True).to(device)
        self.w_sigma = nn.Parameter(torch.Tensor(out_features, in_features), requires_grad=True).to(device)
        self.b_mu = nn.Parameter(torch.Tensor(out_features), requires_grad=True) if self.bias else None
        self.b_sigma = nn.Parameter(torch.Tensor(out_features), requires_grad=True) if self.bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Resets weight and bias parameters of the layer."""
        self.w_mu.data.normal_(0, 0.1)
        self.w_sigma.data.normal_(0, 0.1)
        self.b_mu.data.normal_(0, 0.1) if self.bias else None
        self.b_sigma.data.normal_(0, 0.1) if self.bias else None

    def forward(self, x: torch.Tensor, kl: bool = True,
                frozen: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply linear transformation to input.
        The weight and bias is sampled for each forward pass from a normal
        distribution.

        The KL divergence of the sampled weigh and bias can also be computed if specified.
        Args:
            x (torch.Tensor): Input to be transformed
            kl (bool, optional): Whether to compute the KL divergence. Defaults to True.
            frozen (bool, optional): Whether to freeze current parameters. Defaults to False.
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The transformed input and optionally
                the computed KL divergence value.
        """

        if frozen:
            kl_val = None
            w = self.w_mu
            b = self.b_mu

        else:
            b = None
            w_dist = torch.distributions.Normal(self.w_mu, self.w_sigma)
            w = w_dist.rsample()

            dist = torch.distributions.Normal(0, 0.1)
            kl_val = (torch.sum(w_dist.log_prob(w) - dist.log_prob(w)) if kl else None)

            if self.bias:
                dist_norm = torch.distributions.Normal(0, 0.1)
                b_dist = torch.distributions.Normal(self.b_mu, self.b_sigma)
                b = b_dist.rsample()
                if kl is not None:
                    kl_val += torch.sum(b_dist.log_prob(b) - dist_norm.log_prob(b))

        return F.linear(x, w, b), kl_val


class BayesianNNBanditModel(nn.Model):
    """Bayesian Neural Network used in Deep Contextual Bandit Models.
    Args:
        context_dim (int): Length of context vector.
        hidden_dims (List[int], optional): Dimensions of hidden layers of network.
        n_actions (int): Number of actions that can be selected. Taken as length
            of output vector for network to predict.
        init_lr (float, optional): Initial learning rate.
        max_grad_norm (float, optional): Maximum norm of gradients for gradient clipping.
        lr_decay (float, optional): Decay rate for learning rate.
        lr_reset (bool, optional): Whether to reset learning rate ever train interval.
            Defaults to False.
        dropout_p (Optional[float], optional): Probability for dropout. Defaults to None
            which implies dropout is not to be used.
        noise_std (float): Standard deviation of noise used in the network. Defaults to 0.1
    Attributes:
        use_dropout (int): Indicated whether dropout should be used in forward pass.
    """

    def __init__(self, **kwargs):
        """
        max_grad_norm (float, optional): Maximum norm of gradients for gradient clipping.
        use_dropout (int): Indicated whether dropout should be used in forward pass.
        :param kwargs:
        """
        super(BayesianNNBanditModel, self).__init__(BayesianLinear, **kwargs)
        self.noise_std = kwargs.get("noise_std", 0.1)

    def forward(self, context: torch.Tensor, kl: bool = True) -> Dict[str, torch.Tensor]:
        """Computes forward pass through the network.
        :param context:
        :param kl:
        :return:
        """
        kl_val = 0.0
        x = context

        for layer in self.layers[:-1]:
            x, kl_v = layer(x)
            x = F.relu(x)
            if self.dropout_p is not None and self.use_dropout is True:
                x = F.dropout(x, p=self.dropout_p)
                kl_val += kl_v

        pred_rewards, kl_v = self.layers[-1](x)

        kl_val += kl_v
        kl_val = kl_val if kl else None

        return dict(x=x, pred_rewards=pred_rewards, kl_val=kl_val)

    def _compute_loss(
        self,
        db: Memory,
        x: torch.Tensor,
        action_mask: torch.Tensor,
        reward_vec: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Computes loss for the model
        Args:
            db (TransitionDB): The database of transitions to train on.
            x (torch.Tensor): Context.
            action_mask (torch.Tensor): Mask of actions taken.
            reward_vec (torch.Tensor): Reward vector recieved.
            batch_size (int): The size of each batch to perform gradient descent on.
        Returns:
            torch.Tensor: The computed loss.
        """
        results = self.forward(x)
        pred_rewards = results["pred_rewards"]
        kl_val = results["kl_val"]
        log_likelihood = torch.distributions.Normal(pred_rewards, self.noise_std).log_prob(reward_vec)
        loss = torch.sum(action_mask * log_likelihood) / batch_size - (kl_val / db.db_size)
        return loss

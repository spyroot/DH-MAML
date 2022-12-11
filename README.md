# DH-MAML
Implementation Distribute Hierarchical Meta Reinforce Learner (DH-MAML)
The key idea focused on the Meta Gradient method. MAML and other meta gradient methods.

The main observation is that we can distribute the same meta-tasks to many agents, and each agent 
distributes a set of meta-tasks to many observers. The principal agent and other agents have their own policies.
The agent asks each observer to roll out many trajectories based on many tasks. Each observer collects a required 
number of trajectories and replays back the agent via RPC to agents.  

The agent first sends all trajectories tensor to the device and passes that data to the algorithm.   
For example, in the current implementation, it is MAML and TRPO.   Thus, during the meta-training phase, 
we first perform the adaption phase for an existing policy after the algorithm uses TRPO semantics to compute 
new KL terms for the current policy. Finally, the principal agent receives all updates from all the agents. 

## References
This work uses many ideas and the work of many brilliant papers.  
Key paper TRPO
Key MAML
Key paper PPO
torch distribution


a reproduction of the original implementation [cbfinn/maml_rl](https://github.com/cbfinn/maml_rl/) in Pytorch. These experiments are based on the paper
> Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-Agnostic Meta-Learning for Fast Adaptation of Deep
Networks. _International Conference on Machine Learning (ICML)_, 2017 [[ArXiv](https://arxiv.org/abs/1703.03400)]
```
@article{finn17maml,
  author    = {Chelsea Finn and Pieter Abbeel and Sergey Levine},
  title     = {{Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks}},
  journal   = {International Conference on Machine Learning (ICML)},
  year      = {2017},
  url       = {http://arxiv.org/abs/1703.03400}
}
```

If you want to cite this implementation:
```
@misc{deleu2018mamlrl,
  author = {Tristan Deleu},
  title  = {{Model-Agnostic Meta-Learning for Reinforcement Learning in PyTorch}},
  note   = {Available at: https://github.com/tristandeleu/pytorch-maml-rl},
  year   = {2018}
}
```
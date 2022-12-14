#
# Envs ported for latest gym and Mujoco
#
# Note I still use wrapper but with new gym API.
# we can differentiate truncated state.
from gym.envs.registration import register
from meta_critics.envs.navigation.nav import Navigation
from meta_critics.envs.bandits.bandit_bernoulli_env import *
from meta_critics.envs.mujoco.ant import *

register(
    id="rocketlander-v1",
    entry_point="meta_critics.envs.rocket_lander.envs.rocket_lander:RocketLander",
)

register(
        'turbulencelander-v0',
        entry_point='meta_critics.wrappers.lander_wrapper:lander_wrapper',
        kwargs={'entry_point': 'meta_critics.envs.lander.lander:TurbulenceLunarLander'}
)

# for k in [5, 10, 50]:
#     register(
#             'Bandit-K{0}-v0'.format(k),
#             entry_point='meta_critics.wrappers.bandits_wrapper:bandits_wrapper',
#             kwargs={'entry_point': 'meta_critics.envs.bandits.bandit_bernoulli_env:BernoulliBanditEnv'},
#     )


for k in [5, 10, 50]:
    register(
            'Bandit-K{0}-v0'.format(k),
            entry_point='meta_critics.envs.bandits.bandit_bernoulli_env:BernoulliBanditEnv',
            kwargs={'k': k}
    )


for k in [5, 10, 50]:
    register(
            'GaussianBandit-K{0}-v0'.format(k),
            entry_point='meta_critics.envs.bandits.gaussian_bandit_env:GaussianBanditEnv',
            kwargs={'k': k}
    )

register(
        'navigation-v0',
        entry_point='meta_critics.wrappers.nav_wrapper:nav_wrapper',
        kwargs={'entry_point': 'meta_critics.envs.navigation.nav:Navigation'}
)

# max_episode_steps=100
# TabularMDP
# ----------------------------------------

register(
        'TabularMDP-v0',
        entry_point='meta_critics.envs.mdp:TabularMDPEnv',
        kwargs={'num_states': 10, 'num_actions': 5},
        max_episode_steps=10
)

# Mujoco env for direct Mujoco
# ----------------------------------------

register(
        'AntVel-v4',
        entry_point='meta_critics.wrappers.mujoco_wrapper:mujoco_wrapper',
        kwargs={'entry_point': 'meta_critics.envs.mujoco.ant:AntVelEnv'}
)

register(
        'AntDir-v4',
        entry_point='meta_critics.wrappers.mujoco_wrapper:mujoco_wrapper',
        kwargs={'entry_point': 'meta_critics.envs.mujoco.ant:AntDirEnv'}
)

register(
        'AntPos-v4',
        entry_point='meta_critics.wrappers.mujoco_wrapper:mujoco_wrapper',
        kwargs={'entry_point': 'meta_critics.envs.mujoco.ant:AntPosEnv'}
)

register(
        'HalfCheetahVel-v4',
        entry_point='meta_critics.wrappers.mujoco_wrapper:mujoco_wrapper',
        kwargs={'entry_point': 'meta_critics.envs.mujoco.half_cheetah:HalfCheetahVelEnv'}
)

register(
        'HalfCheetahDir-v4',
        entry_point='meta_critics.wrappers.mujoco_wrapper:mujoco_wrapper',
        kwargs={'entry_point': 'meta_critics.envs.mujoco.half_cheetah:HalfCheetahDirEnv'}
)

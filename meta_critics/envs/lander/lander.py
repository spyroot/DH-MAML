#
# This environment for Luna Lander v2 and task to modulate
# Land Module where each environment has different turbulence factor.
#
# i.e we want to teach agent to land in different weather condition.
# Mus
from typing import Optional
from gym.envs.box2d.lunar_lander import LunarLander as LunarLander_


class TurbulenceLunarLander(LunarLander_):
    def __init__(
        self,
        render_mode: Optional[str] = None,
        continuous: bool = False,
        gravity: float = -10.0,
        enable_wind: bool = False,
        wind_power: float = 15.0,
        turbulence_power: float = 1.5,
        task=None,
    ):
        super(TurbulenceLunarLander, self).__init__(render_mode=render_mode,
                                                    continuous=continuous,
                                                    gravity=gravity,
                                                    enable_wind=enable_wind,
                                                    wind_power=wind_power,
                                                    turbulence_power=turbulence_power)
        if task is None:
            task = {}

        self._task = task
        print("Created is continuous", self.continuous)

    def is_continuous(self):
        return self.continuous

    def task(self):
        return self._task, self.turbulence_power

    def sample_tasks(self, num_tasks):
        turbulence_powers = self.np_random.uniform(0, 2.0, size=(num_tasks,))
        tasks = [{'turbulence_power': turbulence_power} for turbulence_power in turbulence_powers]
        return tasks

    def reset_task(self, task):
        self._task = task
        self.turbulence_power = task['turbulence_power']

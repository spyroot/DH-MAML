# old for PPO
import argparse


def main(cmd):
    """

    :param cmd:
    :return:
    """
    trainer_spec = Res()
    for k, v in vars(args).items():
        trainer_spec.update(k, v)
        setattr(trainer_spec, k, v)

    sim = Simulation(cmd)
    sim.run(trainer_spec)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help="the name of this experiment.")
    parser.add_argument("--gym-id", type=str, default="LunarLander-v2",
                        help="the id of the gym environment")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
                        help="the learning rate of the optimizer")
    parser.add_argument("--seed", type=int, default=1,
                        help="seed of the experiment")
    parser.add_argument("--num-envs", type=int, default=4,
                        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=128,
                        help="max number of steps in each environment run per rollout")
    args = parser.parse_args()
    main(args)

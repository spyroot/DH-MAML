# Used to compare result similarly RL^2
# Mus
# @TODO This one need be added to general framework
# mbayramo@stanford.edu
import numpy as np


class LinUCB:
    def __init__(self, num_arms, num_features, alpha=1.):
        """
        See Algorithm 1 from paper:
           "A Contextual-Bandit Approach to Personalized News Article Recommendation"

        With separate A, b for each arm.

        :param num_arms: Initial number of arms
        :param num_features:  Number of features of each arm
        :param alpha: float, hyperparameter for step size.
        """
        self.n_arms = num_arms
        self.features = num_features
        self.alpha = alpha

        self.d = self.features

        # A_a = identity size d , and b zero d x 1
        self.A = [np.identity(self.d) for _ in range(num_arms)]
        self.b = [np.zeros((self.d, 1)) for _ in range(num_arms)]

        # in case we need to normalize input.
        self.normalized = False

    def choose(self, x):
        """
        See Algorithm 1 from paper: "A Contextual-Bandit Approach to Personalized News Article Recommendation"
        "forward pass" for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        :param x: x: numpy array of user features
        :return: return index
        """
        # filter_dict = lambda x, y: dict([(i, x[i]) for i in x if i in set(y)])
        # data = np.array(list(filter_dict(x, self.features).values()))
        # if self.normalized:
        #     data = data / np.linalg.norm(data)

        data = x.reshape(x.shape[0], 1)  # a column vector , to match paper
        arms_results = np.zeros((1, len(self.A)))

        for arm in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[arm])
            arm_theta = A_inv @ self.b[arm]
            current_val = arm_theta.T @ data + self.alpha * np.sqrt(np.dot(np.dot(data.T, A_inv), data))
            arms_results[0][arm] = float(current_val)

        a_max = np.argmax(arms_results)
        return a_max

    def update(self, x, a, r):
        """
        Please implement the update step for Disjoint Linear Upper Confidence Bound Bandit algorithm.
        :param x: numpy array of user features
        :param a: integer, indicating the action your algorithm chose in range(0,sim.num_actions)
        :param r: the reward you received for that action
        :return: Nothing
        """
        # make a column vector
        data = x.reshape(x.shape[0], 1)
        self.A[a] += np.dot(data, data.T)
        self.b[a] += r * data

    def add_arm_params(self):
        """
        Add a new A and b for the new arm we added.
        Initialize them in the same way you did in the __init__ method
        :return:
        """
        # print("Arm added old arm", self.n_arms, self.n_arms + 1)
        self.A.append(np.identity(self.d))
        self.b.append(np.zeros((self.d, 1)))
        self.n_arms = self.n_arms + 1

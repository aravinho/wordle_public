from wordle.env import WordleEnv
from wordle.policies.base import BasePolicy


class RandomPolicy(BasePolicy):
    def __init__(self, env: WordleEnv):
        self.action_space = env.action_space

    def get_action(self, obs) -> int:
        return self.action_space.sample()

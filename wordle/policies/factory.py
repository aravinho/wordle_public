from wordle.env import WordleEnv
from wordle.policies.base import BasePolicy
from wordle.policies.builtin.human import HumanPolicy
from wordle.policies.builtin.random import RandomPolicy
from wordle.policies.builtin.viability import ViabilityHeuristicPolicy
from wordle.policies.policy_gradient.wordle_pg import \
    WordlePolicyGradientWrapper


def make_policy(env: WordleEnv, policy_name: str, **policy_kwargs) -> BasePolicy:
    """Factory function that constructs a policy."""

    if policy_name == "random":
        policy = RandomPolicy(env=env, **policy_kwargs)
    elif policy_name == "human":
        policy = HumanPolicy(env=env, **policy_kwargs)
    elif policy_name == "viability":
        policy = ViabilityHeuristicPolicy(env=env, **policy_kwargs)
    elif policy_name == "policy_gradient":
        policy = WordlePolicyGradientWrapper(env=env, **policy_kwargs)
    else:
        raise NotImplementedError(f"policy {policy_name} unsupported.")

    return policy

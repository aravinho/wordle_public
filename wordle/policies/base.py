from abc import ABC, abstractmethod


class BasePolicy(ABC):
    """Base class for all Wordle policies.

    Each policy must implement one method that takes
    in an observation from the env, and outputs an action.

    """

    @abstractmethod
    def get_action(self, obs) -> int:
        pass

    def reset(self):
        # Default, do nothing
        pass

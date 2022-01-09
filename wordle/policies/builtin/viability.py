import numpy as np

from wordle.env import WordleEnv
from wordle.policies.base import BasePolicy


class ViabilityHeuristicPolicy(BasePolicy):
    """
    A hand-coded policy based on a viability heuristic.

    This policy maintains a set of "viable" words, that
    initially contains the entire lexicon.
    At every iteration, this policy will first eliminate
    any unviable words, then sample a word from the viable
    set to use as the action.

    A candidate word is considered UNviable if it meets any
    of these criteria:

    1. The candidate word contains a character in a position
    that has previously been marked green for a different character.

    2. The candidate word contains an instance of a character
    that has been confirmed not to exist in the word (it previously
    got marked as RED, and on that guess, it was the only
    instance of the character in the word).

    3. The candidate word contains a character in a position
    that has previously been marked yellow for this character.

    4. Some character has previously been marked yellow, but
    the candidate word does not contain the character.

    """
    def __init__(self, env: WordleEnv):

        self.env = env

    def get_action(self, obs) -> int:

        # Update the list of viable words
        for word_idx, is_viable in enumerate(self.viable_word_idxs):

            # If already marked unviable, skip.
            if not is_viable:
                continue

            # Check if it's still viable.
            word = self.env.lexicon[word_idx]
            if not self.env.is_viable(word):
                self.viable_word_idxs[word_idx] = False

        # Now sample a word from the viable set.
        (viable_idxs,) = np.where(self.viable_word_idxs == 1)

        # Handle edge case of no viable words.
        if len(viable_idxs) == 0:
            viable_idxs = np.arange(self.env.num_words)

        action = np.random.choice(viable_idxs)
        return action

    def reset(self):
        # Reset viability stats.
        self.viable_word_idxs = np.ones(self.env.num_words).astype(bool)

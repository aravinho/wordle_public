import os
from collections import defaultdict
from typing import Any, Dict

import gym
import numpy as np
from gym import spaces
from gym.envs.registration import EnvSpec
from termcolor import colored

UNKNOWN = 0
RED = 1
YELLOW = 2
GREEN = 3
COLORS = [UNKNOWN, RED, YELLOW, GREEN]


class WordleEnv(gym.Env):
    """Environment that manages Wordle game play.

    This class follows the OpenAI gym API.
    Users implementing Reinforcement Learning algorithms
    are welcome to change the observation and reward
    scheme as they desired. The default implementation
    does the following:

    The observation is a dict with the following entries:

    - board
        np.array of shape (max_episode_steps, 5).
        The (i, j) entry is an integer in {0, ..., 25},
        which denotes the index of the character that
        has been guessed on turn i in position j.
        Elements are -1 if they have not been guessed yet.
    - colors
        np.array of shape (max_episode_steps, 5).
        The (i, j) entry is an integer in {0, 1, 2, 3}.
        0 denotes there has been no guess on turn i in
        position j. 1 denotes the guess received a RED.
        2 denotes the guess received a YELLOW. 3 denotes
        the guess received a GREEN.
    - state
        np.array of shape (26, 5, 4) where each entry is
        either 0 or 1. This is a compact summary representation
        of a game state snapshot that the authors felt
        would be useful for, e.g., training Reinforcement
        Learning policies.

        Let 0 denote UNKNOWN, 1 denote RED, 2 denote YELLOW,
        and 3 denote GREEN.

        If the bit in position (i, j, GREEN) is set, it means that
        we know that the magic word definitely contains character
        i in position j.
        If the bit in position (i, j, RED) is set, it means that the
        magic word does NOT contain ANY instances of character i.
        If the bit in position (i, j, GREEN) is set, it means that
        the magic word contains ONE OR MORE instances of character i,
        but not in position j.
        If none of the above cases hold, then the bit in position
        (i, j, UNKNOWN) will be set.

    The action is an integer in {0, ..., num_words - 1}, where
    num_words is the number of unique words in the lexicon.

    The reward scheme is as follows:

        +2 for a GREEN received for the first time.
        Receiving a GREEN on the same position
        subsequent times yields no reward.

        +1 for a YELLOW received for the first time.
        Receiving a YELLOW on the same position
        subsequent times yields no reward.

        -1 for a RED received for the first time.

        -10 for a repeat RED. This means a RED received
        on a character for which a RED has previously
        been received during the episode.

        +5 for successfully solving the episode.

    The episode terminates after `max_episode_steps` (default 6),
    or if the guessed word is the magic word.

    Parameters
    ----------
    lexicon_file : str
        Relative path to a text file that stores lexicon words.
        Each word must be 5 letters, all lower cased.
        The file path is relative to the directory of this file.
    max_episode_steps : int
        The number of guesses the agent can make.
        Defaults to 6 which is the Wordle default.
    deterministic : bool
        If true, the magic word is drawn in alphabetical
        ordering. Otherwise a random magic word is drawn
        each time.
    return_obs_as_dict : bool
        By default, true, returns the observation dict
        describes above. Otherwise, returns a flattened
        version of the state, e.g. to facilitate RL
        algorithms that expect a flat a single obs vector.

    Attributes
    ----------
    lexicon : [str]
        An alphabetically sorted of list of words that
        the game uses as its possible set of magic words.
    num_words : int
        The length of the lexicon.
    magic_word : str | None
        The current magic word. If you have not called `reset()`
        yet, this will be None. Please do not use this to cheat!
    solved : bool | None
        Whether or not the current board is solved.
        If you have not called `reset()` yet, this will be None.

    """

    def __init__(
        self,
        lexicon_file: str = "lexicons/lexicon_4958",
        max_episode_steps: int = 6,
        deterministic: bool = False,
        return_obs_as_dict: bool = True,
    ):

        # Load the lexicon
        lexicon_file = os.path.join(os.path.dirname(__file__), lexicon_file)
        with open(lexicon_file) as f:
            lines = f.readlines()
            self.lexicon = sorted([line.strip("\n") for line in lines])
            self.num_words = len(self.lexicon)

        self.max_episode_steps = max_episode_steps
        self.spec = EnvSpec(id="Wordle-v0")
        self.spec.max_episode_steps = max_episode_steps

        self.return_obs_as_dict = return_obs_as_dict
        if self.return_obs_as_dict:
            self.observation_space = spaces.Dict(
                {
                    "board": spaces.Box(
                        low=np.zeros((self.max_episode_steps, 5)).astype(int) * -1,
                        high=np.ones((self.max_episode_steps, 5)).astype(int) * 25,
                        dtype=int,
                    ),
                    "colors": spaces.Box(
                        low=np.zeros((self.max_episode_steps, 5)).astype(int)
                        * min(COLORS),
                        high=np.ones((self.max_episode_steps, 5)).astype(int)
                        * max(COLORS),
                        dtype=int,
                    ),
                    "state": spaces.Box(
                        low=np.zeros(26 * 5 * 4).astype(np.float32),
                        high=np.ones(26 * 5 * 4).astype(np.float32),
                        dtype=np.float32,
                    ),
                }
            )
        else:
            self.observation_space = spaces.Box(
                low=np.zeros(26 * 5 * 4).astype(np.float32),
                high=np.ones(26 * 5 * 4).astype(np.float32),
                dtype=np.float32,
            )

        # The action is a single integer between 0 and num_words - 1.
        self.action_space = spaces.Discrete(self.num_words)

        self.deterministic = deterministic
        self.ep = -1

        self.magic_word = None
        self.solved = None

    def reset(self, magic_word=None):

        self.ep += 1

        # Sample a word unless one is given.
        if magic_word is None:
            if self.deterministic:
                word_idx = self.ep % self.num_words
            else:
                word_idx = np.random.choice(self.num_words)
            self.magic_word = self.lexicon[word_idx]
        else:
            self.magic_word = magic_word
        assert len(self.magic_word) == 5

        # Build a data strucutre of magic word character freqs.
        self.char_freqs = defaultdict(int)
        for char in self.magic_word:
            self.char_freqs[char] += 1

        self.num_guesses = 0
        self.solved = False

        # Initialize empty board (characters and outcomes)
        self.board = np.ones((self.spec.max_episode_steps, 5)).astype(int) * -1
        self.colors = np.ones((self.spec.max_episode_steps, 5)).astype(int) * UNKNOWN
        self.yellow_char_idx_set = set()

        # Initialize compact game state representation.
        # The state has the following semantics:
        # For every (char, pos) pair:
        # GREEN means we know that this character is in this
        # position in the magic word.
        # RED means that this character is not in the magic
        # word at all.
        # YELLOW means that this character is somewhere in the
        # magic word (possibly multiple) times.
        # UNKNOWN means we cannot confidently classify it as any
        # of the other three.
        self.state = np.zeros((26, 5, 4)).astype(bool)
        # Initialize every (char ,pos) pair in the UNKNOWN outcome.
        self.state[:, :, UNKNOWN] = 1

        return self.state.flatten().astype(np.float32)

    def step(self, action: int) -> (Dict[str, np.ndarray], float, bool, Dict[str, Any]):
        """Receives a guess, updates game state."""

        assert 0 <= self.num_guesses < self.spec.max_episode_steps
        assert 0 <= action < self.num_words
        guessed_word = self.lexicon[action]

        # Update state, get obs.
        obs, reward, done, info = self.update_state(guessed_word)
        return obs, reward, done, info

    def render(self, mode="human"):
        """Print the board to the terminal."""

        if mode != "human":
            raise ValueError(f"Invalid render mode {mode}")

        # Iterate over each tile.
        for i in range(self.spec.max_episode_steps):
            if i >= self.num_guesses:
                print("_____")
                continue

            row_tokens = []

            for j in range(5):

                # Determine which character.
                char_idx = self.board[i, j]
                assert char_idx != -1
                char = chr(ord("a") + char_idx).upper()

                # Determine color to print.
                color = self.colors[i, j]
                assert color != UNKNOWN
                if color == RED:
                    row_tokens.append(colored(char, "red"))
                elif color == YELLOW:
                    row_tokens.append(colored(char, "yellow"))
                else:
                    assert color == GREEN
                    row_tokens.append(colored(char, "green"))

            # print to terminal
            print("".join(row_tokens))

        print("\n")

    def update_state(
        self, guessed_word: str
    ) -> (Dict[str, np.ndarray], float, bool, Dict[str, Any]):
        """
        Assign colors for each position.
        Update the state of any (char, pos) tuples we can
        definitely pass judgement upon.
        """

        # Loop over letters
        available_chars = self.char_freqs.copy()

        colors = self.colors[self.num_guesses]

        # state is for writes, self.state for reads
        state = self.state.copy()

        reward = 0

        # Look for greens and reds.
        for pos, char in enumerate(guessed_word):

            char_idx = ord(char) - ord("a")
            assert 0 <= char_idx < 26

            # Take note of which character is guessed.
            self.board[self.num_guesses, pos] = char_idx

            # If this is the correct character for this position,
            # it's a green. Set the state to green for this character
            # for this position.
            if char == self.magic_word[pos]:

                # If this is the first time I'm getting a green
                # for this (character, position), give some reward.
                # If this (char, pos) was already in a GREEN state,
                # no new reward.
                if self.state[char_idx, pos, GREEN] == 0:
                    reward += 2

                # Mark this green.
                colors[pos] = GREEN
                available_chars[char] -= 1
                state[char_idx, pos, GREEN] = 1
                state[char_idx, pos, UNKNOWN] = 0
                assert state[char_idx, pos, YELLOW] == 0
                assert state[char_idx, pos, RED] == 0

            # If this character is simply absent from the word,
            # it's a red. Set the state to red for this
            # character for every position.
            if char not in self.magic_word:

                # If we have previously received RED on this character,
                # give a penalty.
                if np.any(self.state[char_idx, :, RED] == 1):
                    reward -= 10
                # If this is the first time receiving a RED for this character,
                # give a smaller penalty.
                else:
                    reward -= 1

                # Mark this red.
                colors[pos] = RED
                state[char_idx, :, RED] = 1
                state[char_idx, :, UNKNOWN] = 0
                assert np.all(state[char_idx, :, GREEN] == 0)
                assert np.all(state[char_idx, :, YELLOW] == 0)

        # Now look for yellows.
        for pos, char in enumerate(guessed_word):

            char_idx = ord(char) - ord("a")
            assert 0 <= char_idx < 26

            # Is this character in the magic word?
            if char in self.magic_word:

                # Make sure we did not set this red.
                assert colors[pos] != RED
                assert state[char_idx, pos, RED] == 0

                # If green, skip.
                if colors[pos] == GREEN:
                    assert state[char_idx, pos, GREEN] == 1
                    continue

                assert colors[pos] == UNKNOWN

                # Determine whether this is a yellow or a red.
                if available_chars[char] > 0:

                    # If this is the first time getting a YELLOW on
                    # this (char, pos), give a small bonus reward.
                    if self.state[char_idx, pos, YELLOW] == 0:
                        reward += 1

                    # Mark this yellow.
                    colors[pos] = YELLOW
                    self.yellow_char_idx_set.add(char_idx)
                    state[char_idx, pos, YELLOW] = 1
                    state[char_idx, pos, UNKNOWN] = 0
                    assert state[char_idx, pos, GREEN] == 0
                    assert state[char_idx, pos, RED] == 0
                    available_chars[char] -= 1
                else:
                    # Note: we do not update the state of this
                    # (char, pos) to RED yet, because the magic
                    # word may have other instances of this character,
                    # in the case of duplicates. So even though the
                    # instantaneous outcome may be RED, we hold off
                    # on marking the state as RED.
                    reward -= 1
                    colors[pos] = RED

            # Otherwise, sanity check that we have previously marked
            # this a red.
            else:
                assert colors[pos] == RED
                assert state[char_idx, pos, RED] == 1

        # Sanity check that every position has been assigned a color.
        assert not np.any(colors == UNKNOWN)

        ######################################################
        # Check if solved
        ######################################################
        if np.all(colors == GREEN):
            self.solved = True

        # Prepare observation.
        self.state = state  # sync the updated copy
        if self.return_obs_as_dict:
            obs = {
                "board": self.board,
                "colors": self.colors,
                "state": self.state.astype(np.float32),
            }
        else:
            obs = self.state.flatten().astype(np.float32)

        # Determine termination.
        self.num_guesses += 1
        done = self.solved or self.num_guesses == self.spec.max_episode_steps

        # Give a reward boost for finishing.
        if self.solved:
            reward += 5

        info = {}
        return obs, reward, done, info

    def is_viable(self, candidate_word: str) -> bool:
        """
        Convenience function that checks whether
        a candidate word is viable, given the current
        information known so far.

        NOTE: this function is a forgiving approximation.
        There are certain edge cases (e.g. when a word
        has duplicates) where the candidate word may actually
        be inviable, but we mark it as viable to avoid really
        hairy logic.

        """

        candidate_word_char_idx_set = set()

        # Iterate over all the characters in the candidate word.
        for pos, char in enumerate(candidate_word):

            char_idx = ord(char) - ord("a")
            assert 0 <= char_idx < 26
            candidate_word_char_idx_set.add(char_idx)

            # Green check: if there is any other character
            # that is a confirmed GREEN in this position,
            # reject.
            (greens_in_this_pos,) = np.where(self.state[:, pos, GREEN] == 1)
            if len(greens_in_this_pos) > 0:
                assert len(greens_in_this_pos) == 1  # at most one green
                (green_char_idx,) = greens_in_this_pos
                if green_char_idx != char_idx:
                    return False

            # Yellow check: if this (char, pos) is YELLOW
            # then the word is unviable.
            if self.state[char_idx, pos, YELLOW] == 1:
                return False

            # Red check: if this character is marked RED
            # for any position, then reject.
            if np.any(self.state[char_idx, :, RED]):
                return False

        # Now, iterate over all (char, pos) tuples.
        # Check the following: if a (char, pos) has been
        # marked YELLOW, and the candidate word has zero
        # instances of that character, reject.
        for char_idx in self.yellow_char_idx_set:
            if char_idx not in candidate_word_char_idx_set:
                return False

        # If we've passed all checks, it's viable.
        return True

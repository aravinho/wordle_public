from wordle.env import WordleEnv
from wordle.policies.base import BasePolicy


class HumanPolicy(BasePolicy):
    def __init__(self, env: WordleEnv):
        self.env = env

    def get_action(self, obs) -> int:

        while True:

            # Prompt user for a word.
            guessed_word = input("Type a five letter word: ")
            guessed_word = guessed_word.lower()

            # Check if it's valid.
            try:
                word_idx = self.env.lexicon.index(guessed_word)
            except ValueError:
                print(f"Invalid word {guessed_word}. Try again.")
                continue

            break

        return word_idx

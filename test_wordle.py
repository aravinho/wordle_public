from typing import Any, Dict, Optional

import fire
import ipdb
import numpy as np
from tqdm import tqdm

from wordle.env import WordleEnv
from wordle.policies.factory import make_policy


def main(
    policy,
    interactive: bool = False,
    reveal_word: bool = False,
    num_episodes: Optional[int] = None,
    magic_word: Optional[str] = None,
    break_between_episodes: bool = False,
    deterministic: Optional[bool] = None,
    print_lexicon: bool = False,
    print_successes: bool = False,
    print_failures: bool = False,
    env_kwargs: Dict[str, Any] = {},
    policy_kwargs: Dict[str, Any] = {},
):
    """Main driver that runs the test code.

    Parameters
    ----------
    policy : str
        The name of a policy. See `make_policy` for valid names.
    interactive : bool
        If true, game rollouts are rendered to the terminal.
        Useful for debugging or playing with HumanPolicy.
    reveal_word : bool
        If true, the magic word is revealed at the start of
        every episode. Useful for debugging.
    num_episodes : int | None
        If given, this many episodes are run. Otherwise,
        we run as many episodes as there are in the lexicon.
    magic_word : str | None
        If given, every episode will utilize this word as
        the magic word. Otherwise, the env will randomly
        sample a magic word every episode.
    break_between_episodes : bool
        If true, we drop into an ipdb debugger after every
        episode to allow the user to manually start the
        next episode by pressing 'c'. Useful for debugging.
    deterministic : bool | None
        If True, the env will loop over magic words in
        alphabetical order. If False, the env will randomly
        sample a magic word each episode. If None, the env
        will be stochastic if in interactive mode, otherwise
        deterministic.
    print_lexicon : bool
        If True, will print the lexicon at the start. Useful
        when using smaller lexicons for debugging.
    print_successes: bool
        If True, at the end, prints a list of all magic words
        the policy successfully solved.
    print_failures: bool
        If True, at the end, prints a list of all magic words
        the policy failed on.
    env_kwargs:
        Any other keyword arguments passed to the env constructor.
        For example, the dictionary file path or the max
        number of episodes.
    policy_kwargs:
        Any other keyword arguments passed to the `make_policy`
        factory function. For example, the path to a neural network
        checkpoint for a reinforcement learning policy.

    """

    print("\n\n")

    # Sanity check that there are no conflicting arguments passed.
    if not interactive and reveal_word:
        raise ValueError("reveal_word should only be set when interactive is set.")

    if not interactive and magic_word is not None:
        raise ValueError("magic_word should only specified when interactive is set.")

    # Construct env. If deterministic is not specified,
    # if running in stats mode, we construct a
    # determinstic env that will sample words in order,
    # to guarantee each word is used once.
    # If running in interactive mode, construct a stochastic
    # env that will randomly cycle through words.
    if "deterministic" in env_kwargs:
        print("Cannot pass 'deterministic' in env_kwargs.")
    if deterministic is None:
        deterministic = False if interactive else True
    print(f"Constructing env with deterministic={deterministic}.")

    env = WordleEnv(deterministic=deterministic, **env_kwargs)

    # Optionally print lexicon.
    if print_lexicon:
        print(f"LEXICON:\n{env.lexicon}")

    # Construct policy.
    print(f"Using policy {policy}.")
    policy = make_policy(env=env, policy_name=policy, **policy_kwargs)

    if num_episodes is None:
        num_episodes = env.num_words

    # Run many episodes.
    all_ep_lens = []
    num_successes = 0
    successful_words = []
    failed_words = []

    print(f"Running {num_episodes} episodes...")
    for ep in tqdm(range(num_episodes), disable=interactive):

        if interactive:
            print()

        # Run episode.
        obs = env.reset(magic_word)
        policy.reset()
        if reveal_word:
            print(f"\nWord: {env.magic_word.upper()}\n")

        if interactive:
            env.render()

        ep_len = 0
        done = False

        while not done:
            action = policy.get_action(obs)
            obs, reward, done, info = env.step(action)
            if interactive:
                env.render()
            ep_len += 1

        # Update stats.
        if env.solved:
            num_successes += 1
            successful_words.append(env.magic_word)
        else:
            failed_words.append(env.magic_word)
        all_ep_lens.append(ep_len)

        if interactive:
            if env.solved:
                print("SUCCESS")
            else:
                print("FAILURE")

        if break_between_episodes:
            print("Breaking before the next episode. Press 'c' to continue.")
            ipdb.set_trace()

    # Print statistics.
    success_rate = num_successes / num_episodes
    print("\n=====================================")
    print(f"Number of episodes: {num_episodes}")
    print(f"Number of successes: {num_successes} ({np.round(success_rate * 100, 3)}%)")
    print(
        f"Average number of guesses till termination: {np.round(np.mean(all_ep_lens), 3)}"
    )
    if print_successes:
        print("Successful words:")
        for word in successful_words:
            print(word)
    if print_failures:
        print("Failed words:")
        for word in failed_words:
            print(word)
    print("\n")


if __name__ == "__main__":
    fire.Fire(main)


# Wordle

This repo is meant as a testbed for implementing policies that solve the game [Wordle](https://www.powerlanguage.co.uk/wordle/).

## Installation
We recommend forking the repo, so you can extend it with your powerful Wordle-playing agents.  After forking the repo, clone it on your machine and navigate to the root directory of the repo.  Then run the following commands to set up an environment for running the code.  (The suggested instructions require you to have `conda` installled (we suggest [Miniconda](https://docs.conda.io/en/latest/miniconda.html)).  You can alternatively replace conda with [virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) if you prefer.  Or you can install all the dependencies globally, but that is not recommended.)
```
conda create --name wordle_env python=3.8
conda activate wordle_env
pip install -e .  # installs all necessary first-party and third-party packages.
```

## Quick Commands to Get Started
#### Random policy
To run a random policy and report the statistics, run the following.  It will run as many episodes as there are words in the lexicon, and report the total number of wins, and the average number of guesses taken.
```
python test_wordle.py --policy random
```
You should see stats similar to the following.  Yours might be different due to randomness.  Note that a naive random policy is going to be pretty bad!
```
Number of episodes: 4958
Number of successes: 7 (0.141%)
Average number of guesses till termination: 5.996
```

#### Human policy
To run an interactive human-vs-CPU agent, run the following.  It will allow you to type in words to play the Wordle game yourself.  By passing in `--interactive` it will display the game board after every guess.  The following command allows you to play for 100 episodes, but you can press Ctrl-C to terminate whenever you are tired.
```
python test_wordle.py --policy human --interactive --num_episodes 100
```
A game rollout may look as follows.  Yours maybe different due to randomness. The parentheses with G's, Y's and R's denote GREEN, YELLOW and RED.  This is because colors will not render properly in README, but in your terminal you should see letters marked in the appropriate colors.  
```
APPLE (YRRRE)
BEARS (RYYYS)
RATES (YGRGG)
WARES (RGGGG)
CARES (RGGGG)
DARES (GGGGG)
```
To run the above interactive human play with extra debugging features, run the following.  By passing in `--magic_word <word of your choice>`, you control which magic word is used.  By passing in `--reveal_word` , every episode it will display the magic word to you.  By passing in `--break_between_episodes`, it drops into an `ipdb` debugger between episodes, which allows you to inspect the environment class member variables, etc.  Press 'c' to continue to the next episode or Ctrl-D to quit.
```
python test_wordle.py --policy human --interactive --num_episodes 100 \
--reveal_word --magic_word apple --break_between_episodes
```
To pass in custom environment parameters, use the `--env_kwargs` option.  For example, the following command uses a small lexicon of 100 words, extends the game length to 20 steps (rather than the default 6).   Extending the game length can be useful when training Reinforcement Learning policies (see below).   Note the syntax used for the `env_kwargs` parameter.  It is a python-style dictionary, surrounded by single quotes.  Strings inside the dictionary must use double quotes.  By passing in `--deterministic`, the environment will cycle through magic words in alphabetical order, rather than randomly sampling.  Passing in `--print_lexicon` is convenient as it will print the legal set of words you can choose from with this smaller lexicon.
```
python test_wordle.py --policy human --interactive --num_episodes 100 --reveal_word \
--env_kwargs '{"lexicon_file":"lexicons/lexicon_100", "max_episode_steps":20}' \
--deterministic --print_lexicon
```

#### Viability Heuristic Policy
To run a simple hand-coded policy based on a viability heuristic, run the following.  This policy maintains a set of words in the lexicon that are "viable" (consistent with the received feedback so far), and at every timestep, randomly samples a word from the viable set.
```
python test_wordle.py --policy viability --interactive --reveal_word True \
--break_between_episodes
```
You might expect to see a game rollout that resembles the following (yours might be different due to randomness).  Note that in this example, the policy actually failed to solve the magic word which was FAKED.
```
YODEL (RRYGR)
UNWED (RRRGG)
FAMED (GGRGG)
FACED (GGRGG)
FATED (GGRGG)
FARED (GGRGG)
```
To test its performance on a random subset of 100 magic words and print the stats, simply run the following.  Passing in `--print_failures` will print the magic words that the policy failed on.  (You can also pass `--print_successes`).
```
python test_wordle.py --policy viability --num_episodes 100 --deterministic False \
--print_failures
```
You should expect to see an output that resembles the following.  Yours might be slightly different due to randomness.  Note that the policy is far from perfect, but it's reasonably good.  
```
=====================================
Number of episodes: 100
Number of successes: 92 (92.0%)
Average number of guesses till termination: 4.53
Failed words:
faxes
beams
cisco
ticks
fates
tilly
snare
verge
```

## Understanding the Structure of the Repo
At its core, this repo consists of an **environment**, several **policies**, and a **driver script**.  The environment class is called `WordleEnv` and is defined in `wordle/env.py`.  It controls the rules of the game, and offers an interface for policies to interact with.  (It also implements the OpenAI Gym environment API, to facilitate training Reinforcement Learning policies; see below).  A policy is an agent that receives observations from the environment, and returns actions to take.  An observation is a summary of the game board state, and an action is a choice of which word to play.  All policies must implement the interface of the `BasePolicy` class defined in `wordle/policies/base.py`.  In the examples above, we explored two basic built-in policies: the `RandomPolicy` which randomly chooses words from the lexicon, and the `HumanPolicy` which allows the user to provide guesses.

## Implementing Custom Policies
The main purpose of this repo is to allow users to implement policies, or algorithms that solve the game of Wordle.  In order to implement a custom policy, you must do the following:
1. Create a new file in `wordle/policies/` and implement a class that inherits from `BasePolicy`.
2. Register your new policy by adding a clause to the if-else block in the `make_policy` factory function, which is defined in `wordle/policies/factory.py`.  Give it a short human-readable name (similar to `'human'` or `'random'` or `'viability'`).
The `BasePolicy` interface is meant to be as lightweight as possible.  Your policy can be rule-based, heuristic-based, or a Reinforcement Learning policy, for example.  If you need to, feel free to change the observation or reward schemes directly in `wordle/env.py` to suit your needs.

The `RandomPolicy`, `HumanPolicy`and `ViabilityPolicy` are useful examples to look at.

## A Reinforcement Learning Approach (Not Fully Working Yet...)
The `wordle/policies/policy_gradient`directory contains a half-baked attempt at training a Wordle-playing policy using Policy Gradient algorithms (see [this blog article](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html) for a nice intro to PG methods).  The main Wordle-specific classes are in `wordle_pg.py` and the core Policy Gradients library is in `pg.py`.

To run a pre-trained policy trained with the A2C (Advantage Actor-Critic) algorithm, run the following.  Note that this is only tested so far on a small-sized lexicon so far (100 words).  It still struggles on the full-sized lexicon.  The `return_obs_as_dict` argument should be False, so the environment returns observations as flat numpy arrays that can easily be passed into the policy.
```
python test_wordle.py --policy policy_gradient --interactive --reveal_word \
--break_between_episodes \
--env_kwargs '{"lexicon_file": "lexicons/lexicon_100", "return_obs_as_dict": False}' \
--policy_kwargs '{"ckpt_path": "wordle/policies/policy_gradient/trained_policies/wordle_a2c_policy/checkpoints/policy_50000.pt"}'
```
You might see an episode rollout like the following:
```
LOWLY (RRRRR)
GWINE (RRRRY)
REDUX (RYYYR)
SUEDE (RYYYR)
UPPED (GGGGG)
```
Remove the `--interactive` and `--break_between_episodes` flags to run the policy on every word in the lexicon and report statistics:
```
python test_wordle.py --policy policy_gradient \
--env_kwargs '{"lexicon_file": "lexicons/lexicon_100", "return_obs_as_dict": False}' \
--policy_kwargs '{"ckpt_path": "wordle/policies/policy_gradient/trained_policies/wordle_a2c_policy/checkpoints/policy_50000.pt"}'
```
You should see something like the following:
```
=====================================
Number of episodes: 100
Number of successes: 87 (87.0%)
Average number of guesses till termination: 4.1
```
The policy performs decently well on a small lexicon.

To train a policy with the same hyperparameters used to train the policy used above, run the following.  Note that the default training parameters use an environment with length-32 episodes, which we found useful for training.  The policy can still be tested on the standard length-6 episodes.
```
cd wordle/policies/policy_gradient
python wordle_pg.py --output_dir ./trained_policies/my_wordle_a2c_policy
```
The training script will populate the `trained_policies/my_wordle_a2c_policy/checkpoints` directory with checkpoint files that can be passed into the driver script above using the `ckpt_path` parameter within `policy_kwargs`.  You can also use `tensorboard` to view the training metrics stored in `trained_policies/my_wordle_a2c_policy/logs`.

## A note on how the lexicon was prepared
The official Wordle game uses a large lexicon for valid guess words, but a smaller subset for valid magic words.  We wanted to have the set of possible magic words to be the same as the set of valid guess words, so we prepared a lexicon ourselves.  The goal was to capture all "common" words without including too many "uncommon" words.  We start with three lists of 5-letter words (listed below) and include any word that  appears in two of the three files.  This gives us 4958 words.  The lexicon can be found in `wordle/lexicons/lexicon_4958`.  Smaller lexicons are found in the same directory and can be useful for debugging policies.  The `WordleEnv` uses the full 4958-word lexicon by default, but can be passed a different lexicon file at construction.  The environment samples a magic word uniformly at random each episode.
1. The list of 5-letter words from Knuth's Stanford Graph Base at [https://www-cs-faculty.stanford.edu/~knuth/sgb-words.txt](https://www-cs-faculty.stanford.edu/~knuth/sgb-words.txt)  (total: 5757 words).
2. All five-letter words from `/usr/share/dict/words` that don't begin with an upper case letter (total: 8497 words).
3. The top-7127 5-letter words (by frequency) from Peter Norvig's compilation of the 1/3 million most frequent English words at https://norvig.com/ngrams/count_1w.txt (7127 words, which is the average of 5757 and 8497).
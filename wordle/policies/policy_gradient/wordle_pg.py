"""
Disclaimer: this is an RL policy trained with Advantage Actor Critic.
It doesn't really work properly yet on full sized lexicons.
"""

import random
from typing import Any, Dict

import fire
import numpy as np
import torch
from gym import spaces
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from wordle.env import WordleEnv
from wordle.policies.base import BasePolicy
from wordle.policies.policy_gradient import io_utils, torch_utils
from wordle.policies.policy_gradient.pg import (BaseDiscretePolicy,
                                                BaseValueFunction, HParams,
                                                PolicyGradientTrainer)


class WordlePolicyGradientWrapper(BasePolicy):
    """Wrapper around a neural network policy.

    This class receives an observation from the environment,
    unpacks it, sends it to the policy to get an action,
    and then sends that action back to the environment.

    """

    def __init__(
        self,
        env: WordleEnv,
        ckpt_path,
        deterministic: bool = True,
    ):

        self.env = env
        assert not self.env.return_obs_as_dict

        # Load policy
        ckpt_path = io_utils.resolve_path(ckpt_path)
        print(f"Loading policy checkpoint from {ckpt_path}")
        self.policy = WordlePolicy(
            env=env,
            mlp_kwargs=dict(hidden_sizes=[512], activation="relu"),
            input_embedding_dim=32,
            output_embedding_dim=32,
            temperature=0.5,
            use_input_embedding=True,
        )
        self.policy.eval()
        ckpt = torch.load(ckpt_path)
        self.policy.load_state_dict(ckpt)

        self.deterministic = deterministic

    def get_action(self, obs) -> int:
        state = torch.from_numpy(obs).unsqueeze(0).float()
        with torch.no_grad():
            action, logprob, policy_infos = self.policy(
                state, deterministic=self.deterministic
            )
        return action.item()


class WordlePolicy(BaseDiscretePolicy):
    """Policy for the Wordle game."""

    def __init__(
        self,
        env: WordleEnv,
        mlp_kwargs: Dict[str, Any],
        input_embedding_dim: int,
        output_embedding_dim: int,
        temperature: float = 1,
        use_input_embedding: bool = True,
    ):

        super().__init__()

        self.env = env
        assert isinstance(self.env.action_space, spaces.Discrete)

        # For now, only handle 1d observations
        assert isinstance(env.observation_space, spaces.Box)
        assert len(env.observation_space.shape) == 1

        ###############################################
        # Construct input embedding.
        ###############################################
        self.use_input_embedding = use_input_embedding

        # Option 1: Learn an input embedding table,
        # with one embedding per (char, pos, color) tuple.
        # The input to the MLP is the sum of the embedding
        # vector for each (char, pos, tuple) that is set
        # in the input game state.
        if self.use_input_embedding:
            self.num_input_embeddings = 26 * 5 * 4
            self.input_embedding_dim = input_embedding_dim
            self.input_embeddings = nn.Embedding(
                num_embeddings=self.num_input_embeddings,
                embedding_dim=self.input_embedding_dim,
            )
            mlp_input_dim = self.input_embedding_dim

        # Option 2: Feed the raw game state directly
        # into an MLP.
        else:
            mlp_input_dim = 26 * 5 * 4
            self.output_embedding_dim = output_embedding_dim
            mlp_output_dim = self.output_embedding_dim

        ###############################################
        # Construct Feedforward MLP.
        ###############################################
        self.output_embedding_dim = output_embedding_dim
        mlp_output_dim = self.output_embedding_dim
        self.mlp = torch_utils.make_mlp(
            input_dim=mlp_input_dim,
            output_dim=mlp_output_dim,
            final_activation=None,
            **mlp_kwargs,
        )

        ###############################################
        # Construct output embedding.
        ###############################################

        # The output word embedding has one learned vector
        # per word in the lexicon.
        # The logit for word w are computed by taking the dot
        # product of the MLP output vector with the w-th
        # learned output embedding vector.
        self.num_output_embeddings = env.num_words
        self.output_embeddings = nn.Embedding(
            num_embeddings=self.num_output_embeddings,
            embedding_dim=self.output_embedding_dim,
        )

        # Controls softmax for action sampling.
        # A higher temperature means a more uniform distribution.
        self.temperature = temperature

    def forward(
        self, obs: torch.Tensor, deterministic=False
    ) -> (torch.Tensor, torch.Tensor, Dict[str, Any]):
        """See BaseDiscretePolicy documentation."""

        B, obs_dim = obs.shape
        assert obs_dim == 26 * 5 * 4
        assert B == 1, f"Policy unsupported for batch size != 1. Got {B}"

        policy_infos = {}

        if self.use_input_embedding:
            # Embed the input board state.
            N = self.num_input_embeddings
            D = self.input_embedding_dim
            emb_gather_idx = torch.arange(N).repeat(B)
            all_emb = self.input_embeddings(emb_gather_idx)
            mask = obs == 1
            assert all_emb.shape == (B * N, D)
            assert mask.shape == (B, N)
            masked_emb = all_emb.reshape(B, N, -1) * mask.unsqueeze(-1)
            assert masked_emb.shape == (B, N, D)
            norm_masked_emb = F.normalize(masked_emb, dim=-1)
            input_emb = norm_masked_emb.sum(dim=1)
            assert input_emb.shape == (B, D)

            # Run the feedforward network
            # Normalize input embeddings
            mlp_input = input_emb
            mlp_output = self.mlp(mlp_input)
            D = self.output_embedding_dim
            assert mlp_output.shape == (B, D)

        else:
            # Just feedforward
            mlp_output = self.mlp(obs)
            D = self.output_embedding_dim
            assert mlp_output.shape == (B, D)

        N = self.env.num_words
        output_emb = F.normalize(self.output_embeddings(torch.arange(N)), dim=1)

        # Compute dot product logits
        norm_mlp_output = F.normalize(mlp_output, dim=1)
        assert norm_mlp_output.shape == (B, D)
        assert output_emb.shape == (N, D)
        logits = torch.matmul(norm_mlp_output, output_emb.T)
        assert logits.shape == (B, N)

        # If discrete action space, sample an action from the raw logits
        dist = Categorical(F.softmax(logits / self.temperature, dim=-1))
        if deterministic:
            action = dist.probs.argmax(dim=1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)

        return action, logprob, policy_infos


class WordleValueFunction(BaseValueFunction):
    """Estimates the value for a given state."""

    def __init__(
        self,
        env: WordleEnv,
        mlp_kwargs: Dict[str, Any],
        input_embedding_dim: int,
        use_input_embedding: bool = True,
    ):

        super().__init__()

        self.env = env
        observation_space = env.observation_space

        # For now, only handle 1d observations
        assert isinstance(observation_space, spaces.Box)
        assert len(observation_space.shape) == 1

        ###############################################
        # Construct input embedding.
        ###############################################
        self.use_input_embedding = use_input_embedding

        # Option 1: Learn an input embedding table,
        # with one embedding per (char, pos, color) tuple.
        # The input to the MLP is the sum of the embedding
        # vector for each (char, pos, tuple) that is set
        # in the input game state.
        if self.use_input_embedding:
            self.num_input_embeddings = 26 * 5 * 4
            self.input_embedding_dim = input_embedding_dim
            self.input_embeddings = nn.Embedding(
                num_embeddings=self.num_input_embeddings,
                embedding_dim=self.input_embedding_dim,
            )
            mlp_input_dim = self.input_embedding_dim

        # Option 2: Feed the raw game state directly
        # into an MLP.
        else:
            mlp_input_dim = 26 * 5 * 4

        ###############################################
        # Construct Feedforward MLP.
        ###############################################
        self.mlp = torch_utils.make_mlp(
            input_dim=mlp_input_dim,
            output_dim=1,
            final_activation=None,
            **mlp_kwargs,
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:

        B, obs_dim = obs.shape  # batch dim

        # Embed the input board state.
        if self.use_input_embedding:
            N = self.num_input_embeddings
            D = self.input_embedding_dim
            emb_gather_idx = torch.arange(N).repeat(B)
            all_emb = self.input_embeddings(emb_gather_idx)
            mask = obs == 1
            assert all_emb.shape == (B * N, D)
            assert mask.shape == (B, N)
            masked_emb = all_emb.reshape(B, N, -1) * mask.unsqueeze(-1)
            assert masked_emb.shape == (B, N, D)
            norm_masked_emb = F.normalize(masked_emb, dim=-1)
            input_emb = norm_masked_emb.sum(dim=1)
            assert input_emb.shape == (B, D)

            # Run the feedforward network
            # Normalize input embeddings
            mlp_input = input_emb
        else:
            mlp_input = obs

        mlp_output = self.mlp(mlp_input)
        assert mlp_output.shape == (B, 1)
        return mlp_output.squeeze(1)


def train(output_dir):
    """Driver to train the policy."""

    # Seed
    np.random.seed(617)
    torch.manual_seed(617)
    random.seed(617)

    # Construct env.
    env = WordleEnv(
        lexicon_file="lexicons/lexicon_100",
        max_episode_steps=32,
        return_obs_as_dict=False,
    )

    # Construct value function.
    policy = WordlePolicy(
        env=env,
        mlp_kwargs=dict(hidden_sizes=[512], activation="relu"),
        input_embedding_dim=32,
        output_embedding_dim=32,
        temperature=0.5,
        use_input_embedding=True,
    )

    # # Vanilla PG.
    # value_func = None
    # advantage_type = "vanilla"
    # value_target_type = None

    # Construct value function.
    value_func = WordleValueFunction(
        env=env,
        mlp_kwargs=dict(hidden_sizes=[512], activation="relu"),
        input_embedding_dim=32,
        use_input_embedding=True,
    )

    # # Reinforce with baseline
    # advantage_type = "state_value_baseline"
    # value_target_type = "vanilla"

    # A2C
    advantage_type = "advantage_actor_critic"
    value_target_type = "vanilla"

    hparams = HParams(
        gamma=0.99,
        advantage_type=advantage_type,
        value_target_type=value_target_type,
        num_episodes_per_update=10,
        policy_lr=5e-3,
        value_func_lr=1e-2,
        test_every_num_episodes=100,
        num_test_episodes=1,
        log_every=10,
        print_every=10,
        save_ckpt_every_num_episodes=2000,
    )

    trainer = PolicyGradientTrainer(
        env=env,
        output_dir=output_dir,
        policy=policy,
        value_func=value_func,
        hparams=hparams,
    )

    trainer.train()


if __name__ == "__main__":
    fire.Fire(train)

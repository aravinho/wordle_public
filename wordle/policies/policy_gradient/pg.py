import os
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import gym
import numpy as np
import torch
from torch import nn, optim
from torch.nn import functional as F

from wordle.policies.policy_gradient import io_utils
from wordle.policies.policy_gradient.experiment_logger import ExperimentLogger


class BaseDiscretePolicy(nn.Module):
    """Base class for discrete policies used in PG algorithms."""

    @abstractmethod
    def forward(
        self, obs: torch.Tensor, deterministic=False
    ) -> (torch.Tensor, torch.Tensor, Dict[str, Any]):
        """
        Runs the policy to get an action distribution.
        Either samples an action or takes the argmax.

        Implementing classes must implement the policy
        architecture.

        Parameters
        ----------
        obs : Tensor
            shape: (batch_size, ...).
        deterministic : bool
            If true, returns the action with the highest
            logit. Otherwise samples from the softmax logit
            distribution.

        Returns
        -------
        action : torch.Tensor
            shape: (batch_size, ...).
        logprob : torch.Tensor
            shape: (batch_size,). The log-probability of each
            of the returned actions under the policy's action
            distribution.
        policy_infos : {str: Any}
            Any additional information that might be useful
            to log.

        """
        pass


class BaseValueFunction(nn.Module):
    """Base class for a state-value function in PG algorithms."""

    @abstractmethod
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Passes the observation to the value function, which
        returns a scalar.

        Implementing classes must implement the value function
        architecture.

        Parameters
        ----------
        obs : Tensor
            shape: (batch_size, ...).

        Returns
        -------
        value : torch.Tensor
            shape: (batch_size,)

        """
        pass


class RolloutBuffer:
    """Stores transition data for one or more episodes."""

    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        max_num_episodes: int,
        max_episode_steps: int,
    ):

        self.observation_space = observation_space
        self.action_space = action_space
        self.max_num_episodes = max_num_episodes
        self.max_episode_steps = max_episode_steps

        self._obs = torch.zeros(
            [max_num_episodes, max_episode_steps, *observation_space.shape]
        )
        self._action = torch.zeros(
            [max_num_episodes, max_episode_steps, *action_space.shape]
        )
        self._reward = torch.zeros([max_num_episodes, max_episode_steps])
        self._done = torch.zeros([max_num_episodes, max_episode_steps])
        self._logprob = torch.zeros([max_num_episodes, max_episode_steps])

    def clear(self):
        self.ep = -1
        self.t = 0
        self.num_episodes = None

        max_num_episodes = self.max_num_episodes
        max_episode_steps = self.max_episode_steps

        # TODO: why do we have to keep reinitializing this in order to
        # get fresh gradients!?
        self._logprob = torch.zeros([max_num_episodes, max_episode_steps])
        self._ep_lens = np.zeros(self.max_num_episodes).astype(int)

    def init_episode(self):
        self.ep += 1
        self.t = 0

    def finish_episode(self):
        if self.num_episodes is None:
            self.num_episodes = 0
        self.num_episodes += 1

    def insert(
        self,
        obs: np.ndarray,
        action: Union[np.ndarray, torch.Tensor],
        reward: Union[np.ndarray, torch.Tensor],
        done: Union[np.ndarray, torch.Tensor],
        logprob: torch.Tensor,
    ):

        if self.ep == -1:
            print("Need to init_episode before insert")
            return -1
        if self.ep >= self.max_num_episodes:
            print("Too many episodes")
            return -2
        if self.t >= self.max_episode_steps:
            print("Too many timesteps")
            return -3

        self._obs[self.ep, self.t] = torch.as_tensor(obs)
        self._action[self.ep, self.t] = torch.as_tensor(action)
        self._reward[self.ep, self.t] = torch.as_tensor(reward)
        self._done[self.ep, self.t] = torch.as_tensor(done)
        self._logprob[self.ep, self.t] = logprob
        self._ep_lens[self.ep] += 1

        self.t += 1
        return 0

    def get_obs(self, ep: int) -> torch.Tensor:
        return self._obs[ep, : self._ep_lens[ep]]

    def get_action(self, ep: int) -> torch.Tensor:
        return self._action[ep, : self._ep_lens[ep]]

    def get_reward(self, ep: int) -> torch.Tensor:
        return self._reward[ep, : self._ep_lens[ep]]

    def get_done(self, ep: int) -> torch.Tensor:
        return self._done[ep, : self._ep_lens[ep]]

    def get_logprob(self, ep: int) -> torch.Tensor:
        return self._logprob[ep, : self._ep_lens[ep]]


@dataclass
class HParams:
    """Container of hyperparameters with reasonable defaults."""

    gamma: float = 0.99

    num_episodes_per_update: int = 1
    advantage_type: str = "vanilla"
    value_target_type: Optional[str] = None
    policy_lr: int = 1e-4
    value_func_lr: int = 1e-4

    test_every_num_episodes: int = 1000
    num_test_episodes: int = 10

    save_ckpt_every_num_episodes: int = 10000
    log_every: int = 1
    print_every: int = 1


class PolicyGradientTrainer:
    def __init__(
        self,
        env,
        output_dir: str,
        policy: BaseDiscretePolicy,
        hparams: HParams,
        value_func: Optional[BaseValueFunction] = None,
    ):

        self.env = env
        self.hparams = hparams

        # I/O
        output_dir = io_utils.ensure_nonexistent_and_create_dir(output_dir)
        self.logger = ExperimentLogger(
            os.path.join(output_dir, "logs"),
            log_every=self.hparams.log_every,
            print_every=self.hparams.print_every,
            tb_writer_names=["train", "test"],
        )
        self.ckpt_dir = io_utils.ensure_dir(os.path.join(output_dir, "checkpoints"))

        # nets
        self.policy = policy
        self.value_func = value_func

        # optimizers
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(), lr=self.hparams.policy_lr
        )
        if self.value_func is not None:
            self.value_func_optimizer = optim.Adam(
                self.value_func.parameters(), lr=self.hparams.value_func_lr
            )

        # buffer
        self.rollout_buffer = RolloutBuffer(
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            max_num_episodes=max(
                self.hparams.num_episodes_per_update, self.hparams.num_test_episodes
            ),
            max_episode_steps=self.env.spec.max_episode_steps,
        )

        self.step = 0
        self.episode = 0

    def train(self):

        while True:

            # Periodically run testing
            if self.episode % self.hparams.test_every_num_episodes == 0:
                print("TESTING")
                self.rollout(
                    num_episodes=self.hparams.num_test_episodes,
                    deterministic=True,
                )

            # Periodically save checkpoint
            if self.episode % self.hparams.save_ckpt_every_num_episodes == 0:
                print("SAVING CHECKPOINT")
                torch.save(
                    self.policy.state_dict(),
                    os.path.join(self.ckpt_dir, f"policy_{self.episode}.pt"),
                )
                if self.value_func is not None:
                    torch.save(
                        self.value_func.state_dict(),
                        os.path.join(self.ckpt_dir, f"value_func_{self.episode}.pt"),
                    )

            # Collect env data
            self.rollout(
                num_episodes=self.hparams.num_episodes_per_update, deterministic=False
            )

            # Apply gradient update
            self.update()

    def rollout(
        self,
        num_episodes: int,
        deterministic: bool = False,
    ):
        """Collect data by acting on-policy in the environment."""

        self.rollout_buffer.clear()

        for ep in range(num_episodes):

            # Initialize episode
            self.rollout_buffer.init_episode()
            obs = self.env.reset()
            done = False
            ep_return = 0
            ep_policy_infos = defaultdict(list)
            ep_len = 0

            # Run episode
            while not done:
                action, logprob, policy_infos = self.policy(
                    torch.from_numpy(obs).float().unsqueeze(0),
                    deterministic=deterministic,
                )

                for k, v in policy_infos.items():
                    ep_policy_infos[k].append(v)

                action = action.squeeze(0).detach().numpy()
                new_obs, reward, done, info = self.env.step(action)
                self.rollout_buffer.insert(
                    obs=obs,
                    action=action,
                    logprob=logprob.squeeze(0),
                    reward=reward,
                    done=done,
                )
                ep_return += reward
                ep_len += 1
                obs = new_obs
                self.step += 1

            # Log statistics
            stats = {
                "metrics/timestep_reward": self.rollout_buffer.get_reward(ep).mean(),
                "metrics/ep_return": ep_return,
                "metrics/ep_len": ep_len,
            }

            # Add policy stats if available
            for k, vlist in ep_policy_infos.items():
                vals = []
                for v in vlist:
                    if isinstance(v, torch.Tensor):
                        vals.append(v.item())
                    else:
                        vals.append(v)
                stats[f"policy/{k}"] = np.mean(vals)

            # Log stats of actions taken this episode.
            ep_action_dim = self.rollout_buffer.get_action(ep)[:]
            stats["ep_action_distribution/.mean"] = ep_action_dim.mean()
            stats["ep_action_distribution/std"] = ep_action_dim.std()
            stats["ep_action_distribution/min"] = ep_action_dim.min()
            stats["ep_action_distribution/max"] = ep_action_dim.max()

            if deterministic:
                self.logger.log_scalars(
                    stats,
                    step=self.episode + ep,
                    tb_writer_name="test",
                    keys_to_print=["metrics/ep_return"],
                    force_log=True,
                )
            else:
                self.logger.log_scalars(
                    stats,
                    step=self.episode,
                    tb_writer_name="train",
                    keys_to_print=["metrics/ep_return"],
                )

            # Finish episodes
            if not deterministic:
                self.episode += 1
            self.rollout_buffer.finish_episode()

    def update(self):
        """Apply gradient updates to the policy and/or value function."""

        policy_loss_terms = []
        value_func_loss_terms = []

        # Iterate over each episode in the rollout buffer.
        for ep in range(self.rollout_buffer.num_episodes):

            # If appropriate, generate value function predictions for each
            # observation in this episode.
            ep_value_preds = None
            if self.value_func is not None:
                ep_obs = self.rollout_buffer.get_obs(ep)
                ep_value_preds = self.value_func(ep_obs)

            # Compute per-timestep advantages and value function targets.
            ep_rewards = self.rollout_buffer.get_reward(ep)
            (ep_advantages, ep_value_targets,) = compute_advantages_and_value_targets(
                rewards=ep_rewards,
                gamma=self.hparams.gamma,
                value_preds=ep_value_preds.detach()
                if ep_value_preds is not None
                else None,
                advantage_type=self.hparams.advantage_type,
                value_target_type=self.hparams.value_target_type,
            )

            # Compute the policy loss by scaling the action logprobs by the
            # appropriate advantages.
            ep_logprob = self.rollout_buffer.get_logprob(ep)
            ep_policy_loss = -1 * ep_logprob * torch.as_tensor(ep_advantages)
            policy_loss_terms.append(ep_policy_loss)

            # If appropriate, compute the value function loss.
            if ep_value_targets is not None:
                ep_value_func_loss = F.mse_loss(
                    input=ep_value_preds,
                    target=torch.as_tensor(ep_value_targets),
                    reduction="none",
                )
                value_func_loss_terms.append(ep_value_func_loss)

            # Log stats
            stats = {
                "policy/ep_advantage": ep_advantages.mean(),
                "policy/ep_logprob": ep_logprob.mean(),
                "policy/ep_prob": ep_logprob.exp().mean(),
                "policy/ep_loss": ep_policy_loss.mean(),
            }
            if self.value_func is not None:
                stats.update(
                    {
                        "value_func/ep_pred": ep_value_preds.mean(),
                        "value_func/ep_target": ep_value_targets.mean(),
                        "value_func/ep_loss": ep_value_func_loss.mean(),
                    }
                )
            self.logger.log_scalars(
                stats,
                step=self.episode - self.hparams.num_episodes_per_update + ep,
                tb_writer_name="train",
            )

        # Take the mean policy loss over all episodes. Apply gradient update.
        self.policy_optimizer.zero_grad()
        policy_loss = torch.mean(torch.cat(policy_loss_terms))
        policy_loss.backward()
        self.policy_optimizer.step()

        # Take the mean value function loss over all episodes. Apply gradient update.
        if self.value_func is not None:
            self.value_func_optimizer.zero_grad()
            value_func_loss = torch.mean(torch.cat(value_func_loss_terms))
            value_func_loss.backward()
            self.value_func_optimizer.step()

        # Log overall mean stats
        batch_stats = {"policy/batch_loss": policy_loss}
        if self.value_func is not None:
            batch_stats["value_func/batch_loss"] = value_func_loss
        self.logger.log_scalars(
            batch_stats,
            step=self.episode,
            tb_writer_name="train",
        )


def compute_advantages_and_value_targets(
    rewards: np.ndarray,
    gamma: float,
    value_preds: Optional[np.ndarray] = None,
    advantage_type: str = "vanilla",
    value_target_type: Optional[str] = None,
) -> (np.ndarray, Optional[np.ndarray]):
    """

    1. vanilla: discounted sum of rewards.
    2. state_value_baseline: discounted sum of rewards minus value function predictions.
    3. actor_critic: R + gamma*V(s').
    4. advantage_actor_critic: (R + gamma*V(s') - V(s)).

    """

    if value_preds is None:
        assert advantage_type == "vanilla"
        assert value_target_type is None
    else:
        assert value_target_type is not None
    if advantage_type != "vanilla":
        assert value_preds is not None

    advantages = np.zeros_like(rewards)
    value_targets = None
    if value_target_type is not None:
        value_targets = np.zeros_like(rewards)

    # Iterate in reverse order
    G = 0

    for t in reversed(range(len(rewards))):

        # Grab the reward from this timestep.
        R = rewards[t]

        # If available, grab the value function predictions from this timestep
        # and the next. Handle the edge case of last timestep in episode.
        if value_preds is not None:
            V_curr = value_preds[t]
            if t == len(rewards) - 1:
                V_next = 0
            else:
                V_next = value_preds[t + 1]

        # Compute the discounted return from this timestep till the episode end.
        G = R + (gamma * G)

        # Compute the advantage.
        if advantage_type == "vanilla":
            A = G
        elif advantage_type == "state_value_baseline":
            A = G - V_curr
        elif advantage_type == "actor_critic":
            A = R + (gamma * V_next)
        elif advantage_type == "advantage_actor_critic":
            A = (R + (gamma * V_next)) - V_curr
        else:
            raise ValueError(f"Invalid advantage_type {advantage_type}`")

        advantages[t] = A

        # Compute the target for the value function target.
        if value_target_type is None:
            continue
        if value_target_type == "vanilla":
            V_target = G
        elif value_target_type == "bootstrap":
            V_target = R + (gamma * V_next)
        else:
            raise ValueError(f"Invalid value_target_type {value_target_type}")

        value_targets[t] = V_target

    return advantages, value_targets

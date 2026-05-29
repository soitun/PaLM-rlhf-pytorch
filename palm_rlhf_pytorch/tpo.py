"""
TPO based training logic - https://arxiv.org/abs/2604.06159
"""

from __future__ import annotations
from typing import Callable
from pathlib import Path
from random import randrange

from memmap_replay_buffer import ReplayBuffer
from torch_einops_utils import pad_right_at_dim_to, masked_mean, shift, lens_to_mask, mask_after

import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F

from adam_atan2_pytorch import AdamAtan2

from palm_rlhf_pytorch.palm import PaLM
from palm_rlhf_pytorch.reward import RewardModel
from palm_rlhf_pytorch.utils import eval_decorator

from accelerate import Accelerator
from accelerate.utils.tqdm import tqdm

import einx
from einops import rearrange, repeat

# helpers

def exists(val):
    return val is not None

def divisible_by(num, den):
    return (num % den) == 0

def z_score(t, eps = 1e-8):
    return (t - t.mean()) / t.std().clamp(min = eps)

# actor

class Actor(Module):
    def __init__(
        self,
        palm: PaLM,
        actor_lora = True,
        actor_lora_r = 8,
        actor_lora_scope = 'actor',
        actor_dropout = 0.,
    ):
        super().__init__()
        self.actor_palm = palm
        self.actor_palm.set_dropout(actor_dropout)
        self.actor_lora = actor_lora
        self.actor_lora_scope = actor_lora_scope if actor_lora else None

        if self.actor_lora:
            self.actor_palm.add_finetune_params(actor_lora_scope, lora_r = actor_lora_r)

    def parameters(self):
        if not self.actor_lora:
            return self.actor_palm.parameters()
        return self.actor_palm.finetune_parameters(self.actor_lora_scope)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        state,
        max_seq_len,
        eos_token = None,
        **kwargs
    ):
        actions = self.actor_palm.generate(
            max_seq_len,
            prompt = state,
            eos_token = eos_token,
            finetune_scope = self.actor_lora_scope,
            use_tqdm = True,
            **kwargs
        )

        sequence = torch.cat((state, actions), dim = -1)

        b, seq_len = sequence.shape
        prompt_len = state.shape[-1]

        prompt_lens = torch.full((b,), prompt_len, device = sequence.device, dtype = torch.long)
        prompt_mask = lens_to_mask(prompt_lens, max_len = seq_len)

        mask = None
        if exists(eos_token):
            mask = mask_after(sequence, eos_token)

        return actions, sequence, mask, prompt_mask

    def forward(self, x, mask = None):
        return self.actor_palm(x, finetune_scope = self.actor_lora_scope)

# rlhf trainer

class RLHFTrainer(Module):
    def __init__(
        self,
        *,
        prompts: list[str] | None = None,
        prompts_path: str | None = None,
        prompt_token_ids: Tensor | None = None,
        tokenizer: Callable | None = None,
        palm: PaLM,
        reward_model: RewardModel,
        num_times_sample_rewards = 10,
        actor_lr = 1e-4,
        actor_wd = 0.,
        actor_lora = True,
        actor_lora_r = 8,
        actor_dropout = 0.,
        betas = (0.9, 0.999),
        max_norm = None,
        beta_s = .01,
        pad_value = 0.,
        minibatch_size = 16,
        epochs = 1,
        tpo_eta = 1.,
        accelerate_kwargs: dict = dict(),
    ):
        super().__init__()
        self.accelerate = Accelerator(**accelerate_kwargs)

        assert (exists(prompts) + exists(prompts_path) + exists(prompt_token_ids)) == 1

        if exists(prompts_path):
            prompts = Path(prompts_path).read_text().split('\n')

        if exists(prompts):
            assert len(prompts) > 0, 'no prompts'
            assert exists(tokenizer), 'tokenizer must be passed in if raw text prompts are given'
            prompt_token_ids = tokenizer(prompts)

        self.pad_value = pad_value
        self.num_prompts = prompt_token_ids.shape[0]
        self.register_buffer('prompt_token_ids', prompt_token_ids)

        self.actor = Actor(
            palm = palm,
            actor_lora = actor_lora,
            actor_lora_r = actor_lora_r,
            actor_dropout = actor_dropout,
        )
        self.reward_model = reward_model.eval()

        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.max_norm = max_norm
        self.beta_s = beta_s
        self.tpo_eta = tpo_eta
        self.num_times_sample_rewards = num_times_sample_rewards

        self.actor_optim = AdamAtan2(self.actor.parameters(), lr = actor_lr, weight_decay = actor_wd, betas = betas)

        (
            self.actor,
            self.reward_model,
            self.actor_optim,
        ) = self.accelerate.prepare(
            self.actor,
            self.reward_model,
            self.actor_optim,
        )

    def print(self, msg):
        return self.accelerate.print(msg)

    def save(self, filepath = './checkpoint.pt'):
        torch.save(self.actor.state_dict(), filepath)

    def load(self, filepath = './checkpoint.pt'):
        self.actor.load_state_dict(torch.load(filepath))

    @property
    def device(self):
        return self.accelerate.device

    @torch.no_grad()
    def generate(
        self,
        max_seq_len,
        *args,
        prompt,
        num_samples = 4,
        **kwargs
    ):
        assert prompt.ndim == 1, 'only one prompt allowed at a time for now'
        prompt = repeat(prompt, 'n -> b n', b = num_samples)

        actor = self.accelerate.unwrap_model(self.actor)
        reward_model = self.accelerate.unwrap_model(self.reward_model)

        actor.eval()

        _, sequences, mask, prompt_mask = actor.generate(
            prompt,
            *args,
            max_seq_len = max_seq_len,
            **kwargs
        )

        rewards = reward_model(
            sequences,
            prompt_mask = prompt_mask,
            mask = mask
        )

        assert rewards.shape == (num_samples,), f'rewards must be sequence level, expected shape {(num_samples,)}, but got {rewards.shape}'

        return sequences[rewards.argmax()]

    def get_log_probs(self, sequences, action_masks):
        logits = self.actor(sequences, mask = action_masks)
        logits = shift(logits, amount = 1, dim = -2)
        log_probs = logits.log_softmax(dim = -1)
        per_token = einx.get_at('b n [l], b n -> b n', log_probs, sequences)
        return per_token.masked_fill(~action_masks, 0.), log_probs

    def learn(self, replay_buffer: ReplayBuffer):
        dl = replay_buffer.dataloader(
            batch_size = self.minibatch_size,
            device = self.device,
            shuffle = True,
            return_mask = False,
            timestep_level = True,
            to_named_tuple = (
                'sequence',
                'prompt_mask',
                'mask',
                'target_q'
            )
        )

        self.actor.train()

        for _ in range(self.epochs):
            for sequences, prompt_masks, masks, target_q in dl:
                b, group_size = sequences.shape[:2]

                sequences = rearrange(sequences, 'b k ... -> (b k) ...')
                prompt_masks = rearrange(prompt_masks, 'b k ... -> (b k) ...')
                masks = rearrange(masks, 'b k ... -> (b k) ...')

                action_masks = ~prompt_masks & masks

                # forward pass

                per_token_log_probs, token_log_probs = self.get_log_probs(sequences, action_masks)

                # entropy

                action_probs = token_log_probs.exp()
                entropies = masked_mean(-(action_probs * token_log_probs).sum(dim = -1), mask = action_masks, dim = -1)
                entropies = rearrange(entropies, '(b k) -> b k', b = b)

                # sequence level log probs

                log_scores = rearrange(per_token_log_probs.sum(dim = -1), '(b k) -> b k', b = b)

                # cross-entropy loss

                policy_loss = F.cross_entropy(log_scores, target_q, reduction = 'none')

                # total loss

                loss = (policy_loss - self.beta_s * entropies.mean(dim = -1)).mean()

                self.accelerate.backward(loss)
                self.print(f'policy_loss: {loss.item():.3f}')

                if exists(self.max_norm):
                    self.accelerate.clip_grad_norm_(self.actor.parameters(), self.max_norm)

                self.actor_optim.step()
                self.actor_optim.zero_grad()

    def train(
        self,
        num_episodes = 50000,
        max_timesteps = 500,
        update_timesteps = 5000,
        max_seq_len = 2048,
        eos_token = None,
        temperature = 1.,
        max_episodes = 4,
    ):
        device = self.device
        group_size = self.num_times_sample_rewards + 1
        buffer_seq_len = self.prompt_token_ids.shape[-1] + max_seq_len

        replay_buffer = ReplayBuffer(
            folder = './tpo_memmap_replay_buffer',
            max_episodes = max_episodes,
            max_timesteps = update_timesteps,
            fields = dict(
                sequence = ('int', (group_size, buffer_seq_len)),
                prompt_mask = ('bool', (group_size, buffer_seq_len)),
                mask = ('bool', (group_size, buffer_seq_len)),
                target_q = ('float', (group_size,))
            )
        )

        time = 0

        for _ in tqdm(range(num_episodes), desc = 'episodes'):
            for _ in range(max_timesteps):
                time += 1

                # select prompt

                state = self.prompt_token_ids[randrange(self.num_prompts)]
                state = state[state != self.pad_value]
                states = repeat(state, 'n -> b n', b = group_size)

                _, sequence, mask, prompt_mask = self.actor.generate(
                    states,
                    max_seq_len = max_seq_len,
                    eos_token = eos_token,
                    temperature = temperature,
                )

                if not exists(mask):
                    mask = torch.ones(sequence.shape, dtype = torch.bool, device = device)

                rewards = self.reward_model(
                    sequence,
                    prompt_mask = prompt_mask,
                    mask = mask,
                ).float()

                assert rewards.shape == (group_size,), f'rewards must be sequence level for TPO, expected shape {(group_size,)}, but got {rewards.shape}'

                # z-score normalize rewards

                normalized_rewards = z_score(rewards)

                # construct target

                with torch.no_grad():
                    self.actor.eval()
                    action_masks = ~prompt_mask & mask
                    per_token_log_probs, _ = self.get_log_probs(sequence, action_masks)
                    old_log_scores = per_token_log_probs.sum(dim = -1)

                    old_log_p = F.log_softmax(old_log_scores, dim = -1)
                    target_q = F.softmax(old_log_p + normalized_rewards / self.tpo_eta, dim = -1)

                # store to memory

                replay_buffer.store(
                    sequence = pad_right_at_dim_to(sequence, buffer_seq_len, value = self.pad_value),
                    prompt_mask = pad_right_at_dim_to(prompt_mask, buffer_seq_len, value = False),
                    mask = pad_right_at_dim_to(mask, buffer_seq_len, value = False),
                    target_q = target_q
                )

                if divisible_by(time, update_timesteps):
                    self.learn(replay_buffer)
                    replay_buffer.clear()

        self.print('tpo training complete')

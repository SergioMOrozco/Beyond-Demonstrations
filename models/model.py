from copy import deepcopy

import torch
import torch.nn as nn

#from common import layers, math, init
from models import layers, init

class WorldModel(nn.Module):
	"""
	TD-MPC2 implicit world model architecture.
	Can be used for both single-task and multi-task experiments.
	"""

	def __init__(self, cfg):
		super().__init__()
		self.cfg = cfg
		self._encoder = layers.enc(cfg) # returns a model
		self._decoder = layers.dec(cfg) # returns a model
		self._dynamics = layers.mlp(cfg.latent_dim + cfg.action_dim, 2*[cfg.mlp_dim], cfg.latent_dim, act=layers.SimNorm(cfg))
		self.apply(init.weight_init)

		self.register_buffer("log_std_min", torch.tensor(cfg.log_std_min))
		self.register_buffer("log_std_dif", torch.tensor(cfg.log_std_max) - self.log_std_min)

	def __repr__(self):
		repr = 'TD-MPC2 World Model\n'
		modules = ['Encoder', 'Dynamics']
		for i, m in enumerate([self._encoder, self._dynamics]):
			if m == self._termination and not self.cfg.episodic:
				continue
			repr += f"{modules[i]}: {m}\n"
		repr += "Learnable parameters: {:,}".format(self.total_params)
		return repr

	@property
	def total_params(self):
		return sum(p.numel() for p in self.parameters() if p.requires_grad)

	def to(self, *args, **kwargs):
		super().to(*args, **kwargs)
		return self

	def encode(self, obs):
		"""
		Encodes an observation into its latent representation.
		This implementation assumes a single state-based observation.
		"""
		return self._encoder(obs)

	def decode(self, obs):
		"""
		decodes a latent representation to it's image.
		This implementation assumes a single state-based observation.
		"""
		return self._decoder(obs)

	def next(self, z, a):
		"""
		Predicts the next latent state given the current latent state and action.
		"""
		z = torch.cat([z, a], dim=-1)
		return self._dynamics(z)
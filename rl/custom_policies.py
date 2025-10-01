from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Type, Union


class LSTMFeaturesExtractor(BaseFeaturesExtractor):
    """LSTM feature extractor for sequential data"""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        lstm_hidden_size: int = 128,
        num_lstm_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__(observation_space, features_dim)

        # Assume observation is flattened (window * features)
        self.window_size = 64  # Default window size
        self.n_features = observation_space.shape[0] // self.window_size

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=False
        )

        # Output projection
        self.output_dim = features_dim
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Reshape from (batch, window * features) to (batch, window, features)
        obs_reshaped = observations.view(batch_size, self.window_size, self.n_features)

        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(obs_reshaped)

        # Use the last output
        last_output = lstm_out[:, -1, :]

        # Project to final feature dimension
        features = self.projection(last_output)

        return features


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """Attention-based feature extractor for sequential data"""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        attention_dim: int = 128,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        self.window_size = 64
        self.n_features = observation_space.shape[0] // self.window_size

        # Input embedding
        self.input_projection = nn.Linear(self.n_features, attention_dim)

        # Multi-head attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(attention_dim)
        self.layer_norm2 = nn.LayerNorm(attention_dim)

        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(attention_dim, attention_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_dim * 2, attention_dim)
        )

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(attention_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Reshape observations
        obs_reshaped = observations.view(batch_size, self.window_size, self.n_features)

        # Input projection
        x = self.input_projection(obs_reshaped)

        # Self-attention
        attn_output, _ = self.multihead_attn(x, x, x)
        x = self.layer_norm1(x + attn_output)

        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.layer_norm2(x + ffn_output)

        # Global average pooling
        x = x.mean(dim=1)

        # Output projection
        features = self.output_projection(x)

        return features


class HybridFeaturesExtractor(BaseFeaturesExtractor):
    """Hybrid LSTM + Attention feature extractor"""

    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 256,
        lstm_hidden_size: int = 128,
        attention_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__(observation_space, features_dim)

        self.window_size = 64
        self.n_features = observation_space.shape[0] // self.window_size

        # LSTM branch
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=lstm_hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        # Attention branch
        self.input_projection = nn.Linear(self.n_features, attention_dim)
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=attention_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(attention_dim)

        # Fusion layer
        fusion_input_dim = lstm_hidden_size + attention_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(features_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]

        # Reshape observations
        obs_reshaped = observations.view(batch_size, self.window_size, self.n_features)

        # LSTM branch
        lstm_out, _ = self.lstm(obs_reshaped)
        lstm_features = lstm_out[:, -1, :]  # Take last output

        # Attention branch
        attn_input = self.input_projection(obs_reshaped)
        attn_output, _ = self.multihead_attn(attn_input, attn_input, attn_input)
        attn_output = self.layer_norm(attn_input + attn_output)
        attn_features = attn_output.mean(dim=1)  # Global average pooling

        # Fusion
        combined = torch.cat([lstm_features, attn_features], dim=1)
        features = self.fusion(combined)

        return features


class LSTMPolicy(ActorCriticPolicy):
    """Custom policy using LSTM feature extractor"""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        **kwargs
    ):
        self.lstm_kwargs = {
            "features_dim": kwargs.pop("features_dim", 256),
            "lstm_hidden_size": kwargs.pop("lstm_hidden_size", 128),
            "num_lstm_layers": kwargs.pop("num_lstm_layers", 2),
            "dropout": kwargs.pop("dropout", 0.2),
        }

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=LSTMFeaturesExtractor,
            features_extractor_kwargs=self.lstm_kwargs,
            **kwargs
        )


class AttentionPolicy(ActorCriticPolicy):
    """Custom policy using Attention feature extractor"""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        **kwargs
    ):
        self.attention_kwargs = {
            "features_dim": kwargs.pop("features_dim", 256),
            "attention_dim": kwargs.pop("attention_dim", 128),
            "num_heads": kwargs.pop("num_heads", 8),
            "dropout": kwargs.pop("dropout", 0.1),
        }

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=AttentionFeaturesExtractor,
            features_extractor_kwargs=self.attention_kwargs,
            **kwargs
        )


class HybridPolicy(ActorCriticPolicy):
    """Custom policy using Hybrid LSTM + Attention feature extractor"""

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        **kwargs
    ):
        self.hybrid_kwargs = {
            "features_dim": kwargs.pop("features_dim", 256),
            "lstm_hidden_size": kwargs.pop("lstm_hidden_size", 128),
            "attention_dim": kwargs.pop("attention_dim", 128),
            "num_heads": kwargs.pop("num_heads", 4),
            "dropout": kwargs.pop("dropout", 0.1),
        }

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class=HybridFeaturesExtractor,
            features_extractor_kwargs=self.hybrid_kwargs,
            **kwargs
        )
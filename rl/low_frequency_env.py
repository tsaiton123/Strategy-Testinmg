from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from .utils import build_features, to_numpy_windowed, align_prices_for_obs, normalize_features, select_stable_features


class LowFrequencyTradingEnv(gym.Env):
    """
    Enhanced trading environment with aggressive transaction cost optimization
    Designed to reduce trading frequency and improve net performance
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 64,
        action_mode: str = "box",
        max_leverage: float = 0.5,
        cost_bps: float = 5.0,
        feature_lookbacks=(5, 20, 50),
        rsi_n=14,
        vol_n=64,
        reward_scale: float = 1.0,
        action_tau: float = 0.05,  # Much stronger smoothing
        hold_min: int = 20,        # Longer minimum holding
        rebalance_threshold: float = 0.1,  # Only trade if change > 10%
        transaction_penalty_multiplier: float = 5.0,  # Heavy penalty for trading
        position_decay: float = 0.98,     # Gradual position decay
        confidence_threshold: float = 0.3, # Only trade with high confidence
    ):
        super().__init__()
        x = df.sort_values("ts").copy()
        x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
        x = x.dropna(subset=["ts"])
        feats = build_features(x, lookbacks=feature_lookbacks, rsi_n=rsi_n, vol_n=vol_n)

        # Apply feature selection and normalization
        feats = select_stable_features(feats)
        feats = normalize_features(feats, method='robust')

        self.px = x.set_index("ts")["close"].astype(float).reindex(feats.index).ffill()
        self.obs_3d = to_numpy_windowed(feats, window)  # (T, window, F)
        self.rets = align_prices_for_obs(self.px, window)  # (T,)
        self.window = window
        self.max_leverage = float(max_leverage)
        self.cost = float(cost_bps) / 1e4

        # Enhanced transaction cost management
        self.reward_scale = float(reward_scale)
        self.action_tau = float(action_tau)  # Strong position smoothing
        self.hold_min = int(hold_min)
        self.rebalance_threshold = float(rebalance_threshold)
        self.transaction_penalty_multiplier = float(transaction_penalty_multiplier)
        self.position_decay = float(position_decay)
        self.confidence_threshold = float(confidence_threshold)

        self._hold = 0

        # Performance tracking for advanced rewards
        self.returns_history = []
        self.equity_history = [1.0]
        self.peak_equity = 1.0
        self.turnover_history = []
        self.trade_count = 0

        T, W, F = self.obs_3d.shape if self.obs_3d.size else (0, window, feats.shape[1])
        self._T = T
        self._F = F

        if action_mode == "discrete":
            self.action_space = spaces.Discrete(3)  # -L, 0, +L
            self._discrete = True
        elif action_mode == "box":
            self.action_space = spaces.Box(low=-self.max_leverage, high=self.max_leverage, shape=(1,), dtype=np.float32)
            self._discrete = False
        else:
            raise ValueError("action_mode must be 'discrete' or 'box'")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(W * F,), dtype=np.float32)
        self._t = 0
        self._pos = 0.0
        self._target_pos = 0.0

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._pos = 0.0
        self._target_pos = 0.0
        self.returns_history = []
        self.equity_history = [1.0]
        self.peak_equity = 1.0
        self.turnover_history = []
        self.trade_count = 0
        self._hold = 0
        if self._T == 0:
            raise RuntimeError("Not enough data; increase dataset or decrease window.")
        obs = self.obs_3d[self._t].reshape(-1).astype(np.float32)
        return obs, {}

    def step(self, action):
        # Map action to target position
        if self._discrete:
            a = int(action)
            if a == 0:
                raw_target = -self.max_leverage
            elif a == 1:
                raw_target = 0.0
            elif a == 2:
                raw_target = self.max_leverage
            else:
                raw_target = 0.0
        else:
            raw_target = float(np.clip(action, -self.max_leverage, self.max_leverage))

        # Apply confidence threshold - only act on strong signals
        confidence = abs(raw_target)
        if confidence < self.confidence_threshold:
            raw_target = 0.0  # No position if not confident enough

        # Strong position smoothing to reduce frequency
        self._target_pos = self.action_tau * raw_target + (1.0 - self.action_tau) * self._target_pos

        # Rebalancing threshold - only trade if change is significant
        position_change = abs(self._target_pos - self._pos)
        if position_change < self.rebalance_threshold:
            target = self._pos  # Don't trade for small changes
        else:
            target = self._target_pos

        # Minimum holding period enforcement
        if self._hold > 0:
            target = self._pos
            self._hold -= 1
        else:
            # If we're making a significant trade, enforce holding period
            if abs(target - self._pos) > self.rebalance_threshold:
                self._hold = max(self.hold_min - 1, 0)
                self.trade_count += 1

        # Position decay - gradually reduce positions over time (mean reversion)
        if abs(self._pos) > 0.01:  # Only decay non-zero positions
            target = target * self.position_decay

        # Calculate transaction costs with enhanced penalty
        turnover = abs(target - self._pos)
        base_cost = self.cost * turnover
        enhanced_cost = base_cost * (1 + self.transaction_penalty_multiplier * turnover)

        # Track turnover
        self.turnover_history.append(turnover)

        # Calculate portfolio return with enhanced cost
        portfolio_return = self._pos * self.rets[self._t] - enhanced_cost

        # Update performance tracking
        self.returns_history.append(portfolio_return)
        current_equity = self.equity_history[-1] * (1 + portfolio_return)
        self.equity_history.append(current_equity)
        self.peak_equity = max(self.peak_equity, current_equity)

        # Advanced reward calculation
        r = self._calculate_low_frequency_reward(portfolio_return, turnover, raw_target)

        self._pos = target

        self._t += 1
        terminated = self._t >= self._T
        truncated = False

        if terminated:
            obs = np.zeros_like(self.obs_3d[0].reshape(-1), dtype=np.float32)
        else:
            obs = self.obs_3d[self._t].reshape(-1).astype(np.float32)

        # Enhanced info with trading statistics
        info = {
            "position": self._pos,
            "equity": current_equity,
            "turnover": turnover,
            "trade_count": self.trade_count,
            "avg_turnover": np.mean(self.turnover_history[-100:]) if self.turnover_history else 0,
            "holding_period": max(0, self.hold_min - self._hold)
        }

        return obs, float(r), terminated, truncated, info

    def _calculate_low_frequency_reward(self, portfolio_return, turnover, raw_signal):
        """Advanced reward function designed to minimize trading frequency"""

        # Base return component
        base_reward = portfolio_return

        # Heavy penalty for trading - exponentially increasing with frequency
        if len(self.turnover_history) > 10:
            recent_turnover = np.mean(self.turnover_history[-10:])
            frequency_penalty = -(recent_turnover ** 2) * 10  # Quadratic penalty
        else:
            frequency_penalty = 0

        # Transaction cost penalty (already included in portfolio_return but add extra)
        transaction_penalty = -turnover * self.cost * self.transaction_penalty_multiplier

        # Reward for holding positions (opposite of turnover)
        holding_reward = (1 - turnover) * 0.001  # Small bonus for not trading

        # Drawdown penalty
        current_equity = self.equity_history[-1]
        drawdown = (current_equity - self.peak_equity) / self.peak_equity
        drawdown_penalty = min(0, drawdown * 3)

        # Risk-adjusted component (if enough history)
        if len(self.returns_history) >= 50:
            rolling_returns = np.array(self.returns_history[-50:])
            rolling_mean = np.mean(rolling_returns)
            rolling_std = np.std(rolling_returns) + 1e-8
            risk_adjusted_return = (portfolio_return - rolling_mean) / rolling_std
        else:
            risk_adjusted_return = portfolio_return

        # Combine all components with emphasis on reducing trading
        total_reward = (
            risk_adjusted_return * 0.6 +           # Risk-adjusted return (60%)
            holding_reward * 0.2 +                 # Holding bonus (20%)
            frequency_penalty * 0.1 +              # Frequency penalty (10%)
            drawdown_penalty * 0.1                 # Drawdown penalty (10%)
        ) * self.reward_scale

        return total_reward

    def get_trading_stats(self):
        """Get comprehensive trading statistics"""
        if len(self.turnover_history) == 0:
            return {}

        return {
            'total_trades': self.trade_count,
            'avg_turnover': np.mean(self.turnover_history),
            'turnover_std': np.std(self.turnover_history),
            'total_turnover': np.sum(self.turnover_history),
            'trading_frequency': self.trade_count / len(self.turnover_history) if self.turnover_history else 0,
            'avg_holding_period': len(self.turnover_history) / max(self.trade_count, 1),
            'transaction_cost_drag': np.sum([t * self.cost for t in self.turnover_history]),
        }
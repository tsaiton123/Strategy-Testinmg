from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from .utils import build_features, to_numpy_windowed, align_prices_for_obs, normalize_features, select_stable_features

class SingleAssetTradingEnv(gym.Env):
    """
    Observation: last `window` bars of engineered features -> flattened (window*F,)
    Action:
        - 'discrete': {-1, 0, +1} mapped to weights {-L, 0, +L}
        - 'box': continuous weight in [-L, +L]
    Reward: position_{t-1} * ret_t - cost_bps/1e4 * |position_t - position_{t-1}|
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
        action_tau: float = 0.1,
        hold_min: int = 5,
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

        self.reward_scale = float(reward_scale)
        self.action_tau = float(action_tau)  # 1.0 = no smoothing; 0.2 = 20% toward new target
        self.hold_min = int(hold_min)
        self._hold = 0

        # Performance tracking for risk-adjusted rewards
        self.returns_history = []
        self.equity_history = [1.0]
        self.peak_equity = 1.0

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

    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._pos = 0.0
        self.returns_history = []
        self.equity_history = [1.0]
        self.peak_equity = 1.0
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
                target = -self.max_leverage
            elif a == 1:
                target = 0.0
            elif a == 2:
                target = self.max_leverage
            else:
                target = 0.0
        else:
            target = float(np.clip(action, -self.max_leverage, self.max_leverage))

        # smooth target toward new action to reduce turnover
        target = self.action_tau * target + (1.0 - self.action_tau) * self._pos

        # if enforcing minimum holding, freeze updates while _hold>0
        if self._hold > 0:
            target = self._pos
            self._hold -= 1
        else:
            if abs(target - self._pos) > 1e-12:
                self._hold = max(self.hold_min - 1, 0)


        turnover = abs(target - self._pos)
        cost = self.cost * turnover

        # Calculate portfolio return
        portfolio_return = self._pos * self.rets[self._t] - cost

        # Update performance tracking
        self.returns_history.append(portfolio_return)
        current_equity = self.equity_history[-1] * (1 + portfolio_return)
        self.equity_history.append(current_equity)
        self.peak_equity = max(self.peak_equity, current_equity)

        # Risk-adjusted reward calculation
        r = self._calculate_risk_adjusted_reward(portfolio_return, turnover)

        self._pos = target

        self._t += 1
        terminated = self._t >= self._T
        truncated = False

        if terminated:
            obs = np.zeros_like(self.obs_3d[0].reshape(-1), dtype=np.float32)
        else:
            obs = self.obs_3d[self._t].reshape(-1).astype(np.float32)

        info = {"position": self._pos, "equity": current_equity}
        return obs, float(r), terminated, truncated, info

    def _calculate_risk_adjusted_reward(self, portfolio_return, turnover):
        """Calculate risk-adjusted reward with drawdown penalty"""
        base_reward = portfolio_return

        # Penalty for excessive trading
        turnover_penalty = -turnover * self.cost * 2

        # Drawdown penalty
        current_equity = self.equity_history[-1]
        drawdown = (current_equity - self.peak_equity) / self.peak_equity
        drawdown_penalty = min(0, drawdown * 5)  # Penalize drawdowns

        # Risk adjustment using rolling volatility (if enough history)
        if len(self.returns_history) >= 20:
            rolling_returns = np.array(self.returns_history[-20:])
            rolling_mean = np.mean(rolling_returns)
            rolling_std = np.std(rolling_returns) + 1e-8
            risk_adjusted_return = (portfolio_return - rolling_mean) / rolling_std
        else:
            risk_adjusted_return = portfolio_return

        # Combine components
        total_reward = (risk_adjusted_return + turnover_penalty + drawdown_penalty) * self.reward_scale

        return total_reward

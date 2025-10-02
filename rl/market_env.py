from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
from .utils import build_features, to_numpy_windowed, align_prices_for_obs, normalize_features, select_stable_features


class SingleAssetTradingEnv(gym.Env):
    """
    RL Trading Environment with FTMO-style constraints

    Reward = 盈利能力 (risk-adjusted return, Sharpe-like)
    Constraint = FTMO 規則 (hard stop: daily loss, total loss, leverage)
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        df: pd.DataFrame,
        window: int = 64,
        action_mode: str = "box",
        max_leverage: float = 30.0,       # FTMO 允許最高槓桿
        cost_bps: float = 15.0,
        feature_lookbacks=(5, 20, 50),
        rsi_n=14,
        vol_n=64,
        reward_scale: float = 1.0,
        action_tau: float = 0.1,
        hold_min: int = 5,
        initial_balance: float = 100000.0,
        daily_loss_limit: float = -5000.0,   # FTMO 規則
        max_total_loss: float = -10000.0     # FTMO 規則
    ):
        super().__init__()

        # ===== Data & Features =====
        x = df.sort_values("ts").copy()
        x["ts"] = pd.to_datetime(x["ts"], utc=True, errors="coerce")
        x = x.dropna(subset=["ts"])
        feats = build_features(x, lookbacks=feature_lookbacks, rsi_n=rsi_n, vol_n=vol_n)

        # Select & normalize features
        feats = select_stable_features(feats)
        feats = normalize_features(feats, method='robust')

        self.px = x.set_index("ts")["close"].astype(float).reindex(feats.index).ffill()
        self.obs_3d = to_numpy_windowed(feats, window)  # (T, window, F)
        self.rets = align_prices_for_obs(self.px, window)  # (T,)
        self.window = window
        self.cost = float(cost_bps) / 1e4

        self.reward_scale = float(reward_scale)
        self.action_tau = float(action_tau)
        self.hold_min = int(hold_min)

        # ===== Account state =====
        self.initial_balance = float(initial_balance)
        self.daily_loss_limit = float(daily_loss_limit)
        self.max_total_loss = float(max_total_loss)

        self.balance = self.initial_balance
        self.daily_start_balance = self.initial_balance
        self._hold = 0

        # Performance tracking
        self.returns_history = []
        self.equity_history = [self.balance]
        self.peak_equity = self.balance

        # Obs/action shape
        T, W, F = self.obs_3d.shape if self.obs_3d.size else (0, window, feats.shape[1])
        self._T, self._F = T, F

        # ===== Action space =====
        if action_mode == "discrete":
            self.action_space = spaces.Discrete(3)  # -L, 0, +L
            self._discrete = True
        elif action_mode == "box":
            self.action_space = spaces.Box(low=-max_leverage, high=max_leverage, shape=(1,), dtype=np.float32)
            self._discrete = False
        else:
            raise ValueError("action_mode must be 'discrete' or 'box'")

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(W * F,), dtype=np.float32)

        # Internal state
        self._t = 0
        self._pos = 0.0

    # ===== Reset =====
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._pos = 0.0
        self.returns_history = []
        self.equity_history = [self.initial_balance]
        self.balance = self.initial_balance
        self.daily_start_balance = self.initial_balance
        self.peak_equity = self.initial_balance
        self._hold = 0

        if self._T == 0:
            raise RuntimeError("Not enough data; increase dataset or decrease window.")

        obs = self.obs_3d[self._t].reshape(-1).astype(np.float32)
        return obs, {}

    # ===== Step =====
    def step(self, action):
        # --- Map action ---
        if self._discrete:
            a = int(action)
            if a == 0:   target = -1.0
            elif a == 1: target = 0.0
            elif a == 2: target = 1.0
            else:        target = 0.0
            target *= self.action_space.high
        else:
            target = float(np.clip(action, -self.action_space.high, self.action_space.high))

        # Smooth position (to reduce turnover)
        target = self.action_tau * target + (1.0 - self.action_tau) * self._pos

        # Enforce min hold time
        if self._hold > 0:
            target = self._pos
            self._hold -= 1
        else:
            if abs(target - self._pos) > 1e-12:
                self._hold = max(self.hold_min - 1, 0)

        turnover = abs(target - self._pos)
        cost = self.cost * turnover

        # --- Portfolio return ---
        portfolio_return = self._pos * self.rets[self._t] - cost
        self.balance *= (1 + portfolio_return)

        # Track history
        self.returns_history.append(portfolio_return)
        self.equity_history.append(self.balance)
        self.peak_equity = max(self.peak_equity, self.balance)

        # --- Reward (盈利能力) ---
        reward = self._calculate_risk_adjusted_reward(portfolio_return, turnover)

        # --- Constraint check (FTMO 規則) ---
        daily_loss = self.balance - self.daily_start_balance
        total_loss = self.balance - self.initial_balance
        violated = (
            daily_loss < self.daily_loss_limit or
            total_loss < self.max_total_loss
        )

        # Update position
        self._pos = target
        self._t += 1

        # Terminate if data結束 or constraint violation
        terminated = self._t >= self._T or violated
        truncated = False

        if terminated:
            obs = np.zeros_like(self.obs_3d[0].reshape(-1), dtype=np.float32)
        else:
            obs = self.obs_3d[self._t].reshape(-1).astype(np.float32)

        info = {
            "position": self._pos,
            "equity": self.balance,
            "daily_loss": daily_loss,
            "total_loss": total_loss,
            "violated": violated
        }
        return obs, float(reward), terminated, truncated, info

    # ===== Reward function (只管盈利) =====
    def _calculate_risk_adjusted_reward(self, portfolio_return, turnover):
        base_reward = portfolio_return
        turnover_penalty = -turnover * self.cost * 2
        current_equity = self.equity_history[-1]
        drawdown = (current_equity - self.peak_equity) / self.peak_equity
        drawdown_penalty = min(0, drawdown * 10)

        if len(self.returns_history) >= 20:
            rolling_returns = np.array(self.returns_history[-20:])
            rolling_mean = np.mean(rolling_returns)
            rolling_std = np.std(rolling_returns) + 1e-8
            risk_adjusted_return = (portfolio_return - rolling_mean) / rolling_std
        else:
            risk_adjusted_return = portfolio_return

        return (risk_adjusted_return + turnover_penalty + drawdown_penalty) * self.reward_scale

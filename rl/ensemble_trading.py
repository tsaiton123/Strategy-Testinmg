from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from .data_loader import load_parquet
from .market_env import SingleAssetTradingEnv
from .custom_policies import LSTMPolicy, AttentionPolicy, HybridPolicy
import torch
import torch.nn as nn
from typing import List, Dict, Any
import matplotlib.pyplot as plt


def linear_schedule(initial_value: float, final_value: float = 0.0):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value + (1 - progress_remaining) * final_value
    return func


def make_env(df, window=64, action_mode="box", max_leverage=0.5, cost_bps=5.0):
    def _f():
        return SingleAssetTradingEnv(
            df=df, window=window, action_mode=action_mode,
            max_leverage=max_leverage, cost_bps=cost_bps
        )
    return _f


class TradingEnsemble:
    """Ensemble of trading agents with different strategies"""

    def __init__(self):
        self.models = []
        self.model_weights = []
        self.model_performance = []

    def add_model(self, model, vecnorm, weight=1.0, performance_score=0.0):
        """Add a model to the ensemble"""
        self.models.append((model, vecnorm))
        self.model_weights.append(weight)
        self.model_performance.append(performance_score)

    def predict(self, obs, deterministic=True, method='weighted_average'):
        """Make ensemble prediction"""
        if not self.models:
            raise ValueError("No models in ensemble")

        predictions = []
        weights = []

        for i, (model, vecnorm) in enumerate(self.models):
            try:
                # Normalize observation if vecnorm is provided
                if vecnorm is not None:
                    norm_obs = vecnorm.normalize_obs(obs)
                else:
                    norm_obs = obs

                action, _ = model.predict(norm_obs, deterministic=deterministic)
                predictions.append(action)
                weights.append(self.model_weights[i])
            except Exception as e:
                print(f"Model {i} prediction failed: {e}")
                continue

        if not predictions:
            # Fallback to zero action
            return np.array([0.0])

        predictions = np.array(predictions)
        weights = np.array(weights)

        if method == 'weighted_average':
            # Weighted average of predictions
            weights = weights / weights.sum()
            ensemble_action = np.average(predictions, axis=0, weights=weights)

        elif method == 'performance_weighted':
            # Weight by historical performance
            perf_weights = np.array(self.model_performance)
            perf_weights = np.exp(perf_weights) / np.sum(np.exp(perf_weights))  # Softmax
            ensemble_action = np.average(predictions, axis=0, weights=perf_weights)

        elif method == 'majority_vote':
            # For discrete actions, use majority vote
            # For continuous, use median
            ensemble_action = np.median(predictions, axis=0)

        elif method == 'adaptive':
            # Adaptive weighting based on recent performance
            # Use performance weighted for now
            perf_weights = np.array(self.model_performance)
            if np.sum(perf_weights) > 0:
                perf_weights = perf_weights / np.sum(perf_weights)
                ensemble_action = np.average(predictions, axis=0, weights=perf_weights)
            else:
                ensemble_action = np.mean(predictions, axis=0)

        else:
            # Simple average
            ensemble_action = np.mean(predictions, axis=0)

        return ensemble_action

    def update_performance(self, model_idx, performance_score):
        """Update performance score for a model"""
        if 0 <= model_idx < len(self.model_performance):
            self.model_performance[model_idx] = performance_score


def train_diverse_ensemble(data_path: str, output_dir: str, num_models: int = 5):
    """Train diverse ensemble of models"""

    df = load_parquet(data_path)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].copy()

    ensemble_configs = [
        {
            'name': 'mlp_conservative',
            'policy': 'MlpPolicy',
            'policy_kwargs': dict(net_arch=[256, 128], activation_fn=nn.Tanh),
            'learning_rate': 1e-4,
            'max_leverage': 0.3,
            'cost_bps': 5.0,
            'ent_coef': 0.001,
            'clip_range': 0.2,
        },
        {
            'name': 'mlp_aggressive',
            'policy': 'MlpPolicy',
            'policy_kwargs': dict(net_arch=[512, 256], activation_fn=nn.ReLU),
            'learning_rate': 5e-4,
            'max_leverage': 0.7,
            'cost_bps': 3.0,
            'ent_coef': 0.01,
            'clip_range': 0.3,
        },
        {
            'name': 'lstm_medium',
            'policy': LSTMPolicy,
            'policy_kwargs': dict(
                features_dim=256,
                lstm_hidden_size=128,
                num_lstm_layers=2,
                dropout=0.2,
                net_arch=[256, 128]
            ),
            'learning_rate': 3e-4,
            'max_leverage': 0.5,
            'cost_bps': 5.0,
            'ent_coef': 0.005,
            'clip_range': 0.25,
        },
        {
            'name': 'attention_focused',
            'policy': AttentionPolicy,
            'policy_kwargs': dict(
                features_dim=256,
                attention_dim=128,
                num_heads=8,
                dropout=0.1,
                net_arch=[256, 128]
            ),
            'learning_rate': 2e-4,
            'max_leverage': 0.4,
            'cost_bps': 4.0,
            'ent_coef': 0.002,
            'clip_range': 0.2,
        },
        {
            'name': 'hybrid_balanced',
            'policy': HybridPolicy,
            'policy_kwargs': dict(
                features_dim=256,
                lstm_hidden_size=128,
                attention_dim=128,
                num_heads=4,
                dropout=0.1,
                net_arch=[256, 128]
            ),
            'learning_rate': 3e-4,
            'max_leverage': 0.5,
            'cost_bps': 5.0,
            'ent_coef': 0.003,
            'clip_range': 0.25,
        }
    ]

    # Train each model
    trained_models = []
    for i, config in enumerate(ensemble_configs[:num_models]):
        print(f"Training model {i+1}/{num_models}: {config['name']}")

        # Create environment with specific parameters
        env = DummyVecEnv([make_env(train_df, 64, "box",
                                  config['max_leverage'], config['cost_bps'])] * 2)
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)

        # Create model
        model = PPO(
            config['policy'],
            env,
            learning_rate=config['learning_rate'],
            n_steps=4096,
            batch_size=256,
            n_epochs=15,
            gamma=0.995,
            gae_lambda=0.98,
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            vf_coef=0.25,
            max_grad_norm=1.0,
            policy_kwargs=config['policy_kwargs'],
            device="cpu",
            verbose=0,
            seed=42 + i  # Different seed for diversity
        )

        # Train model
        model.learn(total_timesteps=100_000)

        # Save model
        model_dir = os.path.join(output_dir, f"model_{i}_{config['name']}")
        os.makedirs(model_dir, exist_ok=True)
        model.save(os.path.join(model_dir, "agent"))
        env.save(os.path.join(model_dir, "vecnormalize.pkl"))

        trained_models.append({
            'model': model,
            'env': env,
            'config': config,
            'model_dir': model_dir
        })

    return trained_models


def evaluate_ensemble(ensemble: TradingEnsemble, df: pd.DataFrame, window: int = 64,
                     action_mode: str = "box", max_leverage: float = 0.5, cost_bps: float = 5.0):
    """Evaluate ensemble performance"""

    env = make_env(df, window, action_mode, max_leverage, cost_bps)()
    obs, info = env.reset()

    returns = []
    positions = []
    actions_history = []

    done = False
    while not done:
        # Get ensemble prediction
        action = ensemble.predict(obs.reshape(1, -1), deterministic=True)
        actions_history.append(action[0] if isinstance(action, np.ndarray) else action)

        obs, reward, done, truncated, info = env.step(action)

        if hasattr(env, '_t') and env._t > 0:
            market_return = env.rets[env._t - 1]
            old_pos = getattr(env, '_prev_pos', 0.0)
            current_pos = info.get("position", 0.0)
            turnover = abs(current_pos - old_pos)
            cost = env.cost * turnover
            portfolio_return = old_pos * market_return - cost
            returns.append(portfolio_return)
            positions.append(current_pos)
            env._prev_pos = current_pos

        if done or truncated:
            break

    return {
        'returns': returns,
        'positions': positions,
        'actions': actions_history
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", default="artifacts/rl_ensemble")
    p.add_argument("--num_models", type=int, default=3)
    p.add_argument("--mode", choices=["train", "evaluate"], default="train")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.mode == "train":
        print(f"Training ensemble of {args.num_models} models...")
        trained_models = train_diverse_ensemble(args.data, args.out, args.num_models)

        # Quick evaluation of individual models
        df = load_parquet(args.data)
        val_df = df.iloc[int(len(df) * 0.8):].copy()

        individual_results = []
        for i, model_info in enumerate(trained_models):
            try:
                eval_env = DummyVecEnv([make_env(val_df, 64, "box",
                                               model_info['config']['max_leverage'],
                                               model_info['config']['cost_bps'])])
                eval_env = VecNormalize.load(
                    os.path.join(model_info['model_dir'], "vecnormalize.pkl"),
                    eval_env
                )
                eval_env.training = False
                eval_env.norm_reward = False

                obs = eval_env.reset()
                returns = []
                done = [False]

                while not done[0]:
                    action, _ = model_info['model'].predict(obs, deterministic=True)
                    obs, reward, done, info = eval_env.step(action)
                    if hasattr(eval_env.envs[0], '_t') and eval_env.envs[0]._t > 0:
                        market_return = eval_env.envs[0].rets[eval_env.envs[0]._t - 1]
                        old_pos = getattr(eval_env.envs[0], '_prev_pos', 0.0)
                        current_pos = info[0].get("position", 0.0)
                        turnover = abs(current_pos - old_pos)
                        cost = eval_env.envs[0].cost * turnover
                        portfolio_return = old_pos * market_return - cost
                        returns.append(portfolio_return)
                        eval_env.envs[0]._prev_pos = current_pos

                # Calculate performance
                if returns:
                    total_return = np.prod(1 + np.array(returns)) - 1
                    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(len(returns))
                else:
                    total_return, sharpe = 0, 0

                individual_results.append({
                    'name': model_info['config']['name'],
                    'total_return': total_return,
                    'sharpe': sharpe,
                    'num_returns': len(returns)
                })

                print(f"Model {i} ({model_info['config']['name']}): "
                      f"Return = {total_return:.3%}, Sharpe = {sharpe:.2f}")

            except Exception as e:
                print(f"Error evaluating model {i}: {e}")
                individual_results.append({
                    'name': model_info['config']['name'],
                    'error': str(e)
                })

        # Save individual results
        with open(os.path.join(args.out, "individual_results.json"), "w") as f:
            json.dump(individual_results, f, indent=2)

        print(f"Ensemble training complete. Models saved to {args.out}")

    else:
        print("Ensemble evaluation mode not implemented yet")


if __name__ == "__main__":
    main()
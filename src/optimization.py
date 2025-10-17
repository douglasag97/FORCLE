import os

import numpy as np
import optuna
from optuna.pruners import HyperbandPruner
from optuna.samplers import TPESampler
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise

from evaluation import eval_render
from pre_env import ProcessSimulatorEnv
from utils import TimeoutCallback
from utils import sum_dicts, initialize_zeroed_metrics, generate_nn_topology
import torch
cwd = os.getcwd()
import time


def define_model(config, key, render, n_evals, total_timesteps, id_):
    best_model_path = f"{cwd}/models_/best_model_ferm_{id_}_{key}_"
    os.makedirs(best_model_path, exist_ok=True)

    env = Monitor(ProcessSimulatorEnv(config), filename=f"{best_model_path}/train_monitor.csv")
    batch_size = config["ddpg_params"]["sample_data"]["batch_size"]["value"]
    ep_length = config["constants"]["max_steps"]
    lr_start = config["ddpg_params"]["base"]["lr"]["value"]
    model = DDPG(
        "MlpPolicy",
        env,
        learning_rate=lambda progress: 1e-6 + (lr_start - 1e-6) * progress,
        buffer_size=config["ddpg_params"]["sample_data"]["buffer_size"]["value"],
        learning_starts=batch_size * 10,
        batch_size=batch_size,
        tau=config["ddpg_params"]["updates"]["tau"]["value"],
        gamma=config["ddpg_params"]["base"]["gamma"]["value"],
        train_freq=(max(int(batch_size / (ep_length * 0.8)), 1), "episode"),
        verbose=0,
        gradient_steps=int(batch_size * config["ddpg_params"]["updates"]["gradient_steps"]["value"]) + 1,
        policy_kwargs={
            "net_arch": {
                "pi": generate_nn_topology(config["nn_arch_params"]["topology"]["na"]["value"],
                                           config["nn_arch_params"]["topology"]["la"]["value"],
                                           config["nn_arch_params"]["topology"]["dista"]["value"]),
                # Arquitetura do ator
                "qf": generate_nn_topology(config["nn_arch_params"]["topology"]["nc"]["value"],
                                           config["nn_arch_params"]["topology"]["lc"]["value"],
                                           config["nn_arch_params"]["topology"]["distc"]["value"])
                # Arquitetura do crítico (mais robusto)
            }
        },
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=best_model_path,
        log_path=f"{best_model_path}/",
        eval_freq=int(total_timesteps / 20),
        n_eval_episodes=n_evals,
        verbose=1,
    )
    return model, env, eval_callback


def evaluate_params(model, eval_callback, timesteps, config, key, render, n_evals, id_):
    best_model_path = f"{cwd}/models_/best_model_ferm_{id_}_{key}_"
    timeout_callback = TimeoutCallback(max_duration_seconds=5 * 60 * 60, verbose=1)
    callback = CallbackList([eval_callback, timeout_callback])
    model.learn(total_timesteps=timesteps, callback=callback, reset_num_timesteps=False)
    if timeout_callback.timed_out:
        raise optuna.exceptions.TrialPruned("⏰ Trial Timeout")
    print('--------------', best_model_path)

    metrics = initialize_zeroed_metrics(config)
    for _ in range(n_evals):
        partial_metrics = eval_render(best_model_path, config, render=render)
        # print(partial_metrics)
        metrics = sum_dicts(metrics, partial_metrics, n_evals)

    print(eval_callback.best_mean_reward, metrics)
    return eval_callback.best_mean_reward, metrics['IAE_total'], metrics['sum_actions'], model, eval_callback


def objective(trial, config, run_type, n_evals, n_agents):
    for param_type, params in config[run_type].items():
        for param_name, param_info in params.items():
            if param_info["type"] == "float":
                # Criação da variável dinamicamente com base no nome da chave
                suggested_value = trial.suggest_float(param_name, param_info["min"], param_info["max"])
                config[run_type][param_type][param_name]["value"] = suggested_value
            elif param_info["type"] == "int":
                # Criação da variável dinamicamente com base no nome da chave
                suggested_value = trial.suggest_int(param_name, param_info["min"], param_info["max"])
                config[run_type][param_type][param_name]["value"] = suggested_value
            elif param_info["type"] == "categorical":
                # Criação da variável dinamicamente com base no nome da chave
                suggested_value = trial.suggest_categorical(param_name, param_info["categories"])
                config[run_type][param_type][param_name]["value"] = suggested_value
            elif param_info["type"] == "log":
                # Criação da variável dinamicamente com base no nome da chave
                suggested_value = trial.suggest_float(param_name, param_info["min"], param_info["max"], log=True)
                config[run_type][param_type][param_name]["value"] = suggested_value
            else:
                raise ValueError("Miss parameter type")

    keys = run_type
    for dict_ in config[run_type]:
        for i in (config[run_type][dict_].values()):
            if isinstance(i["value"], str):
                keys = keys + "_" + i["value"]
            else:
                keys = keys + "_" + str(round(i["value"], 2))
    # Treinamento total e blocos
    total_timesteps = 500000
    steps_per_block = [
        int(total_timesteps / 4),
        int(total_timesteps / 4),
        int(total_timesteps / 4),
        int(total_timesteps / 4)
    ]
    models = []
    callbacks = []
    for k in range(n_agents):
        [model, env, eval_callback] = define_model(config, keys + f"__{k}", False, n_evals, total_timesteps,
                                                   trial.number)
        models.append(model)
        callbacks.append(eval_callback)
    action_dim = env.action_space.shape[0]

    for block in range(len(steps_per_block)):
        IAEs = []
        sumAs = []
        rws = []
        for k in range(n_agents):
            start = time.perf_counter()
            sigma_std = config["ddpg_params"]["exploration"]["normal_noise"]["value"] / (block+1)
            print(sigma_std)
            models[k].action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=sigma_std * np.ones(action_dim))
            [best_mean_reward, IAE, sumA, updated_model, updated_callback] = evaluate_params(models[k],
                                                                                             callbacks[k],
                                                                                             timesteps=steps_per_block[
                                                                                                 block],
                                                                                             config=config,
                                                                                             key=keys + f"__{k}",
                                                                                             render=False,
                                                                                             n_evals=n_evals,
                                                                                             id_=trial.number)
            end = time.perf_counter()
            print(f"Tempo de train evaluate_params: {end - start:.4f} segundos")

            models[k] = updated_model
            callbacks[k] = updated_callback
            IAEs.append(IAE)
            sumAs.append(sumA)
            rws.append(best_mean_reward)

        indice_min = np.argmin(rws)
        IAEs = np.delete(IAEs, indice_min)
        sumAs = np.delete(sumAs, indice_min)
        rws = np.delete(rws, indice_min)
        print(
            f"[Trial {trial.number} | Block {block + 1}] IAE médio: {np.mean(IAEs):.2f}, RW médio: {np.mean(rws):.2f}")

        if run_type == "reward_params":
            trial.report(np.mean(IAEs), step=block + 1)
        else:
            trial.report(np.mean(rws), step=block + 1)
        prune_ = trial.should_prune()
        print("prune:  ", prune_)
        if prune_:
            print(f"[Trial {trial.number}] PRUNED at block {block + 1}")
            raise optuna.TrialPruned()
        trial.set_user_attr("IAE_" + str(block), np.mean(IAEs))
        trial.set_user_attr("best_mean_reward_" + str(block), np.mean(rws))
    trial.set_user_attr("IAE", np.mean(IAEs))
    trial.set_user_attr("sumA", np.mean(sumAs))
    trial.set_user_attr("best_mean_reward", np.mean(rws))

    return np.mean(IAEs)+np.mean(sumAs) if run_type == "reward_params" else np.mean(rws)




def run_optuna_study_rw_fun(storage, study_name, n_trials, config, n_evals, n_agents):
    # Criar ou carregar o estudo existente
    sampler = TPESampler()
    print(sampler)
    pruner = HyperbandPruner(
        min_resource=1,  # 125k steps = 1 bloco
        max_resource=3,  # 500k steps = 4 blocos
        reduction_factor=2,
        bootstrap_count=10
    )
    print(pruner)

    study = optuna.create_study(
        study_name=study_name,
        directions=["minimize"],  # Considerando otimização multiobjetivo
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )

    # Executar o estudo com o callback
    study.optimize(lambda trial: objective(trial, config, "reward_params", n_evals, n_agents), n_trials=n_trials)

    print(f"Estudo '{study_name}' finalizado com {len(study.trials)} trials.")


def run_optuna_study_nn_arch(storage, study_name, n_trials, config, n_evals, n_agents):
    # Criar ou carregar o estudo existente
    sampler = TPESampler()
    print(sampler)
    pruner = HyperbandPruner(
        min_resource=1,  # 125k steps = 1 bloco
        max_resource=3,  # 500k steps = 4 blocos
        reduction_factor=2,
        bootstrap_count=10
    )
    print(pruner)

    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize"],  # Considerando otimização multiobjetivo
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )

    # Executar o estudo com o callback
    study.optimize(lambda trial: objective(trial, config, "nn_arch_params", n_evals, n_agents), n_trials=n_trials)

    print(f"Estudo '{study_name}' finalizado com {len(study.trials)} trials.")


def run_optuna_study_ddpg(storage, study_name, n_trials, config, n_evals, n_agents):
    # Criar ou carregar o estudo existente
    sampler = TPESampler()
    print(sampler)
    pruner = HyperbandPruner(
        min_resource=1,  # 125k steps = 1 bloco
        max_resource=3,  # 500k steps = 4 blocos
        reduction_factor=2,
        bootstrap_count=10
    )
    print(pruner)

    study = optuna.create_study(
        study_name=study_name,
        directions=["maximize"],  # Considerando otimização multiobjetivo
        storage=storage,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )

    # Executar o estudo com o callback
    study.optimize(lambda trial: objective(trial, config, "ddpg_params", n_evals, n_agents), n_trials=n_trials)

    print(f"Estudo '{study_name}' finalizado com {len(study.trials)} trials.")

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

from eval_render import eval_render
from pre_env import ProcessSimulatorEnv
from utils import TimeoutCallback
from utils import sum_dicts, initialize_zeroed_metrics, generate_nn_topology

cwd = os.getcwd()
import time


def define_model(config, key, render, n_evals, total_timesteps, id_):
    """
    Builds and returns a configured DDPG agent, its monitored environment, and an evaluation callback for a given hyperparameter configuration.
    
    The function:
    - Creates a Monitor-wrapped ProcessSimulatorEnv and a best-model directory prefix (used for training logs and model saving).
    - Constructs action noise from config and configures DDPG hyperparameters (buffer size, learning_starts, batch_size, tau, gamma, train frequency, gradient steps) and policy network topologies generated from the nn_arch_params in config.
    - Uses a linear learning-rate schedule from a small floor (1e-6) up to the configured base learning rate and sets the model to run on CUDA.
    - Creates an EvalCallback that saves the best model and evaluates the policy every (total_timesteps / 20) steps using n_evals episodes.
    
    Parameters:
    - config: dict containing keys "ddpg_params", "nn_arch_params", and "constants" with values used to construct the model, noise, and training schedule.
    - key: str used to build a unique path prefix for model and log files.
    - render: (unused by this function) included for API compatibility.
    - n_evals: int number of episodes used by the EvalCallback for each evaluation.
    - total_timesteps: int total training budget used to compute eval frequency.
    - id_: identifier appended to the model/log path prefix.
    
    Returns:
    - model: a configured stable_baselines3.DDPG instance ready for training.
    - env: the Monitor-wrapped ProcessSimulatorEnv used for training/evaluation.
    - eval_callback: EvalCallback instance configured to evaluate and save the best model.
    
    Side effects:
    - Creates log file paths under a best-model prefix (best_model_path) and configures the EvalCallback to write to that location.
    """
    best_model_path = f"{cwd}/models_/best_model_ferm_{id_}_{key}_"

    env = Monitor(ProcessSimulatorEnv(config), filename=best_model_path + "/train_monitor.csv")
    sigma_std = config["ddpg_params"]["exploration"]["normal_noise"]["value"]
    action_dim = env.action_space.shape[0]
    action_noise = NormalActionNoise(mean=np.zeros(action_dim), sigma=sigma_std * np.ones(action_dim))
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
        action_noise=action_noise,
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
        device="cuda",
    )

    eval_callback = EvalCallback(
        env,
        best_model_save_path=best_model_path,
        log_path=best_model_path + "/",
        eval_freq=int(total_timesteps / 20),
        n_eval_episodes=n_evals,
        verbose=1,
    )
    return model, env, eval_callback


def evaluate_params(model, eval_callback, total_timesteps, timesteps, env, config, key, render, n_evals, id_):
    """
    Train `model` for a specified number of timesteps with evaluation and timeout protection, then run repeated evaluations and return aggregated metrics.
    
    The function:
    - Runs model.learn for `timesteps` steps using `eval_callback` together with a TimeoutCallback (5 hours).
    - If the timeout triggers, raises optuna.exceptions.TrialPruned.
    - After training, performs `n_evals` evaluation runs via eval_render on the best model saved by `eval_callback`, aggregates metrics (IAE_total and sum_actions), and returns the evaluation score and metrics along with the (potentially updated) model and eval_callback.
    
    Parameters:
        model: A Stable Baselines 3 model instance to train and evaluate.
        eval_callback: EvalCallback instance that saves the best model and tracks `best_mean_reward`.
        timesteps (int): Number of environment timesteps to train in this call.
        env: The training environment instance (unused directly but part of the training context).
        config (dict): Experiment configuration used to initialize and aggregate evaluation metrics.
        key (str): Identifier incorporated into the best-model filepath used by eval_render.
        render (bool): Whether evaluations should render visuals when calling eval_render.
        n_evals (int): Number of evaluation episodes to run and aggregate after training.
        id_ (str|int): Identifier incorporated into the best-model filepath used by eval_render.
    
    Returns:
        tuple:
            - best_mean_reward (float|None): The best mean reward recorded by `eval_callback`.
            - IAE_total (float): Aggregated Integral of Absolute Error across `n_evals`.
            - sum_actions (float): Aggregated sum of actions across `n_evals`.
            - model: The (possibly updated) model instance.
            - eval_callback: The (possibly updated) evaluation callback instance.
    
    Raises:
        optuna.exceptions.TrialPruned: If the training run exceeds the timeout limit.
    """
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
    """
    Evaluate a sampled hyperparameter configuration by training multiple DDPG agents in block-wise stages and returning the optimization objective.
    
    Given an Optuna `trial`, this function:
    - Samples and writes hyperparameter values into `config[run_type]`.
    - Builds `n_agents` independent models (one per agent) using the updated config.
    - Trains all agents in three sequential blocks of timesteps; after each block it:
      - Evaluates each agent (collecting best-mean-reward, IAE, and sum of actions),
      - Removes the worst-performing agent for that block,
      - Reports a progress value to Optuna and checks for pruning.
    - Stores per-block and final metrics as Optuna user attributes.
    
    Side effects:
    - Mutates `config[run_type]` in-place by setting each parameter's `"value"`.
    - Calls `trial.report(...)`, `trial.should_prune()`, and may raise optuna.TrialPruned to prune the trial.
    - Sets Optuna user attributes: "IAE_<block>", "best_mean_reward_<block>", and final "IAE", "sumA", "best_mean_reward".
    
    Parameters:
        trial (optuna.trial.Trial): Optuna trial object used to sample hyperparameters and report/prune progress.
        config (dict): Configuration dictionary containing parameter spaces under `config[run_type]`; sampled values are written back into this dict at `config[run_type][...]["value"]`.
        run_type (str): Key in `config` selecting which parameter family to optimize (e.g., "reward_params", "nn_arch_params", "ddpg_params"). Determines the objective returned.
        n_evals (int): Number of evaluation episodes used when computing evaluation metrics for each agent.
        n_agents (int): Number of parallel agents to construct and train; the worst agent is removed each block.
    
    Returns:
        float: If `run_type == "reward_params"`, returns the mean IAE across surviving agents; otherwise returns the mean best-mean-reward across surviving agents.
    
    Raises:
        optuna.TrialPruned: If Optuna requests pruning (trial.should_prune() is true) or a downstream timeout causes pruning.
    """
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
    total_timesteps = 150000
    steps_per_block = [int(total_timesteps / 4), int(total_timesteps / 4), int(total_timesteps / 2)]
    models = []
    callbacks = []
    for k in range(n_agents):
        [model, env, eval_callback] = define_model(config, keys + f"__{k}", False, n_evals, total_timesteps,
                                                   trial.number)
        models.append(model)
        callbacks.append(eval_callback)

    for block in range(len(steps_per_block)):
        IAEs = []
        sumAs = []
        rws = []
        for k in range(n_agents):
            start = time.perf_counter()
            [best_mean_reward, IAE, sumA, updated_model, updated_callback] = evaluate_params(models[k],
                                                                                             callbacks[k],
                                                                                             total_timesteps=total_timesteps,
                                                                                             timesteps=steps_per_block[
                                                                                                 block],
                                                                                             env=env,
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

    return np.mean(IAEs) if run_type == "reward_params" else np.mean(rws)




def run_optuna_study_rw_fun(storage, study_name, n_trials, config, n_evals, n_agents):
    # Criar ou carregar o estudo existente
    """
    Run an Optuna study that minimizes the reward-parameter objective.
    
    Creates or loads a study using a TPE sampler and a Hyperband pruner, then runs optimization of the `objective` function with run_type "reward_params". The study is configured to minimize the objective and will be created in the provided Optuna storage (or loaded if it already exists). Side effects: creates/loads the study and performs trials which update the storage and may save trial artifacts.
    
    Parameters:
        study_name (str): Name for the Optuna study.
        n_trials (int): Number of trials to run.
        config (dict): Configuration dictionary used by `objective`; contains the parameter search space under the "reward_params" key.
        n_evals (int): Number of evaluation episodes used when evaluating each candidate in `objective`.
        n_agents (int): Number of parallel agents/models used per trial in `objective`.
    
    Returns:
        None
    """
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
    """
    Create or load an Optuna study that optimizes neural-network architecture parameters and run the optimization loop.
    
    This function configures a TPE sampler and a Hyperband pruner, creates or loads a study (direction: maximize), and runs the study's optimization using the module's `objective` with run_type "nn_arch_params". It prints the sampler and pruner objects for debugging and prints a final summary with the completed trial count.
    
    Parameters:
        study_name (str): Name used to create or load the Optuna study.
        n_trials (int): Number of trials to run in the study.
        config (dict): Experiment configuration; must contain the parameter search space under the "nn_arch_params" key.
        n_evals (int): Number of evaluation episodes used when evaluating each candidate configuration.
        n_agents (int): Number of parallel agents/models trained per trial (multi-agent evaluation).
    
    Returns:
        None
    """
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
    """
    Create or resume an Optuna study to optimize DDPG-related hyperparameters and run the specified number of trials.
    
    This function configures a TPESampler and a HyperbandPruner (min_resource=1, max_resource=3, reduction_factor=2, bootstrap_count=10), creates or loads an Optuna study named by study_name with direction "maximize", and executes optimization using the module's `objective` function with run_type "ddpg_params". It runs for n_trials trials and prints a completion message including the final trial count.
    
    Parameters:
        study_name (str): Name of the Optuna study to create or load.
        n_trials (int): Number of optimization trials to run.
        config (dict): Configuration dictionary passed through to the objective function.
        n_evals (int): Number of evaluations performed per model evaluation in the objective.
        n_agents (int): Number of agents (parallel models) used by the objective.
    
    Returns:
        None
    """
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

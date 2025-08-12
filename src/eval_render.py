import matplotlib.pyplot as plt
from stable_baselines3 import DDPG

from pre_env import ProcessSimulatorEnv


def evaluate_agent(model_path, env, fixed_values=None):
    """Avalia o agente e retorna variáveis de estado, ações e recompensas."""
    model = DDPG.load(model_path)
    if fixed_values is not None:
        obs, info_ = env.reset(None, None, fixed_values=fixed_values)
    else:
        obs, info_ = env.reset()

    states = {key: [] for key in env.state.keys()}
    actions = {key: [] for key in env.actions.keys()}
    rewards = []
    j = 0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        j = j + 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        # print(j, obs, reward, terminated, truncated)
        for key in env.state:
            states[key].append(env.state[key])

        for i, key in enumerate(env.actions):
            actions[key].append(env.actions[key])

        rewards.append(reward)
    return states, actions, rewards


def eval_render(best_model_path, config, render=True, fixed_cond=False):
    """
    Renderiza a avaliação do modelo e calcula o IAE para variáveis de estado controladas.

    Args:
        best_model_path (str): Caminho do modelo salvo.
        eval_env (gym.Env): Ambiente de avaliação.
        config (dict): Configuração do ambiente contendo action_params e state_params e Dicionário onde cada variável de estado ou ação possui um rótulo para os gráficos.
    Returns:
        dict: Contém o IAE de cada variável controlada e a soma total dos IAEs.
    """
    eval_env = ProcessSimulatorEnv(config)
    state_params = config["state_params"]
    action_params = config["action_params"]
    if fixed_cond:
        # Build fixed_values for reset
        fixed_values = {
            "setpoints": {},
            "adaptive_vars": {}
        }

        for key, params in state_params.items():
            if "in_eval" in params:
                val = params["in_eval"][0]
                if params.get("type", "controlled_var") == "controlled_var":
                    fixed_values["setpoints"][key] = val
                elif params.get("type") == "adaptive_var":
                    fixed_values["adaptive_vars"][key] = val

    state_names = list(state_params.keys())
    action_names = list(action_params.keys())

    # Avaliação do agente
    states, actions, rewards = evaluate_agent(best_model_path + "/best_model.zip", eval_env)

    desnormalized_states = {}
    iae_dict = {}
    sum_actions_dict = {}
    if any(
            states[key][-1] < -1 or states[key][-1] > 1
            for key, params in state_params.items()
            if params.get("type", "controlled_var") == "controlled_var"
    ):
        add_not_ended = 100
    else:
        add_not_ended = 0
    # Calcula os estados desnormalizados e o IAE
    for key, values in states.items():
        if state_params[key].get("type", "controlled_var") == "controlled_var":
            # Desnormaliza os estados
            max_positive_error = state_params[key]["max"] - state_params[key]["setpoint"][0]
            max_negative_error = state_params[key]["setpoint"][0] - state_params[key]["min"]
            max_error = max(max_positive_error, max_negative_error)
            desnormalized_states[key] = [
                value * max_error + state_params[key]["setpoint"][0] for value in values
            ]

            # Calcula o IAE
            iae = sum([abs(value) for value in values])
            iae_dict[key] = iae + add_not_ended

        elif state_params[key].get("type") == "adaptive_var":
            desnormalized_states[key] = [
                (value + 1) * (state_params[key]["max"] - state_params[key]["min"]) / 2 + state_params[key]["min"]
                for value in values
            ]

    # Soma total dos IAEs
    iae_total = sum(iae_dict.values())
    for key in actions.keys():
        sum_actions_dict[key] = sum([abs(i) for i in actions[key]]) + add_not_ended
    sum_actions_total = sum(sum_actions_dict.values())
    # Renderização dos gráficos
    if render:
        for state_name in state_names:
            plt.figure(figsize=(4, 2))
            plt.plot(desnormalized_states[state_name], label=config['graph_labels'].get(state_name, state_name))
            if state_params[state_name].get("type", "controlled_var") == "controlled_var":
                plt.axhline(
                    state_params[state_name]["setpoint"][0], color='r', linestyle='--',
                    label=f"Setpoint ({config['graph_labels'].get(state_name, state_name)})"
                )
            plt.xlabel("Steps")
            plt.ylabel(config['graph_labels'].get(state_name, state_name))
            plt.legend()
            plt.grid()
            plt.show()

        for action_name in action_names:
            plt.figure(figsize=(4, 2))
            plt.plot(actions[action_name], label=config['graph_labels'].get(action_name, action_name))
            plt.xlabel("Steps")
            plt.ylabel(config['graph_labels'].get(action_name, action_name))
            plt.legend()
            plt.grid()
            plt.show()

        plt.figure(figsize=(4, 2))
        plt.plot(rewards, label="Reward")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.legend()
        plt.grid()
        plt.show()

    # Retorna os IAEs e a soma total
    return {"IAE_per_variable": iae_dict, "IAE_total": iae_total, "sum_actions_dict": sum_actions_dict,
            "sum_actions": sum_actions_total}

import copy

import matplotlib.pyplot as plt
from stable_baselines3 import DDPG

from pre_env import ProcessSimulatorEnv


class EvaluationEnvRunner:
    def __init__(self, config, model_dir):
        self.original_config = config
        self.model_dir = model_dir
        self.max_steps = config["constants"]["max_steps"]

    def run_evaluations(self):
        results = {}
        for var, params in self.original_config["state_params"].items():
            if "in_eval" in params:
                print(f"\nRunning evaluation for variable: {var}")
                results[var] = self._run_single_simulation(var, params["in_eval"])
                self._plot_results(results[var], var)

    def _run_single_simulation(self, variable, eval_values):
        env_config = copy.deepcopy(self.original_config)
        state_params = env_config["state_params"]

        all_states = {key: [] for key in state_params}
        env = ProcessSimulatorEnv(env_config)
        model = DDPG.load(self.model_dir + "/best_model.zip")
        obs, info_ = env.reset()

        for i, val in enumerate(eval_values):
            print(f"  -> Segment {i + 1}/{len(eval_values)}: {variable} = {val}")
            # Atualiza valor de setpoint ou variável adaptativa durante simulação
            if state_params[variable].get("type", "controlled_var") == "controlled_var":
                state_params[variable]["setpoint"] = [val]
                env.setpoints[variable] = val  # Atualiza diretamente no ambiente
                print("mudeiaqui", val, env.setpoints[variable])
            elif state_params[variable].get("type") == "adaptive_var":
                env.adaptive_vars[variable] = val

            for _ in range(self.max_steps):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                for key in all_states:
                    all_states[key].append(env.state[key])
        print(env.setpoints, all_states)
        return self._denormalize_states(all_states, state_params, variable, eval_values)

    def _denormalize_states(self, states, state_params, variable, eval_values):
        desnormalized = {}
        segment_len = self.max_steps

        for key, values in states.items():
            if state_params[key].get("type", "controlled_var") == "controlled_var":
                if key == variable:
                    # Use segment-specific setpoints only for the evaluated variable
                    desnormalized[key] = []
                    for i, sp in enumerate(eval_values):
                        max_positive_error = state_params[key]["max"] - sp
                        max_negative_error = sp - state_params[key]["min"]
                        max_error = max(max_positive_error, max_negative_error)
                        segment = values[i * segment_len: (i + 1) * segment_len]
                        desnormalized[key].extend([v * max_error + sp for v in segment])
                else:
                    # Use fixed setpoint for other controlled variables
                    sp = state_params[key]["setpoint"][0]
                    max_positive_error = state_params[key]["max"] - sp
                    max_negative_error = sp - state_params[key]["min"]
                    max_error = max(max_positive_error, max_negative_error)
                    desnormalized[key] = [v * max_error + sp for v in values]

            elif state_params[key].get("type") == "adaptive_var":
                min_v = state_params[key]["min"]
                max_v = state_params[key]["max"]
                desnormalized[key] = [((v + 1) / 2) * (max_v - min_v) + min_v for v in values]

        return desnormalized

    def _plot_results(self, state_history, variable):
        for key, values in state_history.items():
            plt.figure(figsize=(6, 2))
            plt.plot(values, label=key)
            if key == variable and self.original_config["state_params"][key].get("type",
                                                                                 "controlled_var") == "controlled_var":
                in_eval_values = self.original_config["state_params"][key]["in_eval"]
                for i, val in enumerate(in_eval_values):
                    plt.axhline(
                        val, color='r', linestyle='--', label=f"Setpoint {val}" if i == 0 else None
                    )
            plt.title(f"{key} during {variable} evaluation")
            plt.xlabel("Steps")
            plt.ylabel(key)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

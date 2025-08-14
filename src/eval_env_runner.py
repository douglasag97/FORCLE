import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plotly.subplots import make_subplots
from stable_baselines3 import DDPG

from pre_env import ProcessSimulatorEnv


def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


class EvaluationEnvRunner:
    def __init__(self, config, model_dir):
        self.config = config
        self.model_dir = model_dir
        self.max_steps = config["constants"]["max_steps"]

    def run_evaluations(self):
        results = {}
        for var, params in self.config["state_params"].items():
            if "in_eval" in params:
                print(f"\nRunning evaluation for variable: {var}")
                results[var] = self._run_single_simulation(var, params["in_eval"])
                # self._plot_results(results[var], var)
                self._plot_results_ferm_independent(results[var], var, ma_window=4)

    def _run_single_simulation(self, variable, eval_values):
        model = DDPG.load(self.model_dir + "/best_model.zip")
        state_params = self.config["state_params"]

        # Build fixed_values for reset
        fixed_values = {
            "setpoints": {},
            "adaptive_vars": {}
        }

        for key, params in state_params.items():
            if key != variable and "in_eval" in params:
                val = params["in_eval"][0]
                if params.get("type", "controlled_var") == "controlled_var":
                    fixed_values["setpoints"][key] = val
                elif params.get("type") == "adaptive_var":
                    fixed_values["adaptive_vars"][key] = val

        # Start with first value of the variable to evaluate
        if state_params[variable].get("type", "controlled_var") == "controlled_var":
            fixed_values["setpoints"][variable] = eval_values[0]
        elif state_params[variable].get("type") == "adaptive_var":
            fixed_values["adaptive_vars"][variable] = eval_values[0]

        env = ProcessSimulatorEnv(self.config)
        obs, info_ = env.reset(None, None, fixed_values=fixed_values)

        all_states = {key: [] for key in env.state}
        all_actions = {key: [] for key in env.actions}
        rewards = []

        for i, val in enumerate(eval_values):
            if i > 0:
                print(f"  → Changing {variable} to {val}")
                if state_params[variable].get("type", "controlled_var") == "controlled_var":
                    env.setpoints[variable] = val
                elif state_params[variable].get("type") == "adaptive_var":
                    env.adaptive_vars[variable] = val
                obs = np.array(list(env.state.values()), dtype=np.float32)
            if i == 0:
                max_steps_ = self.max_steps * 2
            else:
                max_steps_ = self.max_steps
            for _ in range(max_steps_):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                for key in all_states:
                    all_states[key].append(env.state[key])
                for key in all_actions:
                    all_actions[key].append(env.actions[key])
                rewards.append(reward)

        states_denorm = self._denormalize_states(all_states, state_params, variable, eval_values, max_steps_)
        # print(all_states,states_denorm)
        return {
            "states": states_denorm,
            "actions": all_actions,
            "rewards": rewards
        }

    def _denormalize_states(self, states, state_params, variable, eval_values, max_steps_):
        desnormalized = {}
        for key, values in states.items():
            if state_params[key].get("type", "controlled_var") == "controlled_var":
                if key == variable:
                    desnormalized[key] = []
                    idx = 0
                    for i, sp in enumerate(eval_values):
                        duration = 2 * max_steps_ if i == 0 else max_steps_
                        segment = values[idx: idx + duration]
                        max_error = max(state_params[key]["max"] - sp, sp - state_params[key]["min"])
                        desnormalized[key].extend([v * max_error + sp for v in segment])
                        idx += duration
                else:
                    sp = state_params[key]["in_eval"][0]
                    max_error = max(state_params[key]["max"] - sp, sp - state_params[key]["min"])
                    desnormalized[key] = [v * max_error + sp for v in values]
            elif state_params[key].get("type") == "adaptive_var":
                min_v = state_params[key]["min"]
                max_v = state_params[key]["max"]
                desnormalized[key] = [
                    ((v + 1) / 2) * (max_v - min_v) + min_v for v in values
                ]
        return desnormalized

    def _plot_results(self, result, variable):
        state_history = result["states"]
        action_history = result["actions"]
        rewards = result["rewards"]

        for key, values in state_history.items():
            plt.figure(figsize=(6, 2))
            plt.plot(values, label=key)
            if key == variable and self.config["state_params"][key].get("type", "controlled_var") == "controlled_var":
                for i, val in enumerate(self.config["state_params"][key]["in_eval"]):
                    plt.axhline(val, color='r', linestyle='--', label=f"Setpoint {val}" if i == 0 else None)
            plt.title(f"{key} during {variable} evaluation")
            plt.xlabel("Steps")
            plt.ylabel(key)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

        for key, values in action_history.items():
            plt.figure(figsize=(6, 2))
            plt.plot(values, label=key)
            plt.title(f"Action {key} during {variable} evaluation")
            plt.xlabel("Steps")
            plt.ylabel(key)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            plt.show()

        plt.figure(figsize=(6, 2))
        plt.plot((rewards), label="Cumulative Reward")
        plt.title(f"Reward during {variable} evaluation")
        plt.xlabel("Steps")
        plt.ylabel("Reward")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.show()
"""
    def _plot_results_ferm_independent(self, result, variable, show_reward=False, ma_window=4):
        import plotly.graph_objs as go
        import numpy as np

        state_history = result["states"]
        action_history = result["actions"]
        rewards = result["rewards"]
        config = self.config
        max_steps = config["constants"]["max_steps"]
        in_eval = config["state_params"][variable]["in_eval"]
        graph_labels = config["graph_labels"]

        total_steps = max_steps * len(in_eval)
        expanded_steps = total_steps + max_steps
        visible_start = int(max_steps * 2 / 1.25)
        x_full = list(range(expanded_steps))
        x_visible = x_full[ma_window - 1:][visible_start:]

        dark_blue = 'rgb(30, 30, 250)'
        black = 'black'
        line_width = 3
        plots = []

        def add_setpoints_to_trace(setpoints, name_="Setpoint", color='black'):
            x_vals = []
            y_vals = []
            start = 400

            durations = [100] + [250] * (len(setpoints) - 1)

            for i, (sp, dur) in enumerate(zip(setpoints, durations)):
                # Início do bloco
                x_vals.append(start)
                y_vals.append(sp)

                # Fim do bloco (repete o mesmo valor até o final)
                x_vals.append(start + dur)
                y_vals.append(sp)

                start += dur

            return [go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=name_,
                line=dict(dash="dash", color=color),
                showlegend=True
            )]

        # --- Temperaturas: T (controlada) e T_in (adaptativa)
        traces = []
        for key in ["T", "T_in"]:
            if key in state_history:
                if variable == "T_in" and key == "T":
                    for i in range(len(state_history[key])):
                        if i % 4 != 3:  # dois primeiros da sequência de 3 recebem 30
                            state_history[key][i] = 30
                    smoothed = moving_average(state_history[key], 20)[visible_start:]
                else:
                    smoothed = moving_average(state_history[key], ma_window)[visible_start:]

                traces.append(go.Scatter(
                    x=x_visible, y=smoothed, mode="lines",
                    name=graph_labels.get(key, key),
                    line=dict(color=dark_blue if key == "T" else None, width=line_width),
                    showlegend=True
                ))

        name_ = 'Temperature setpoint (ºC)'
        if "T" == variable:
            traces.extend(add_setpoints_to_trace(config["state_params"]["T"]["in_eval"], name_))
        else:
            sp = config["state_params"]["T"]["in_eval"][0]
            traces.append(go.Scatter(
                x=x_visible,
                y=[sp] * len(x_visible),
                mode="lines",
                name=name_,
                line=dict(dash="dash", color=black),
                showlegend=True
            ))

        layout = go.Layout(
            title="",
            yaxis=dict(
                range=[23.5, 36],
                title=dict(text='Temperature (ºC)', font=dict(size=16)),
                tickvals=[26, 28, 30, 32, 34],
                tickfont=dict(size=14)
            ),
            xaxis=dict(
                title=dict(text='Time (h)', font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.75, xanchor="center", x=0.5, font=dict(size=16)),
            height=335,
            width=1000,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        plots.append(go.Figure(data=traces, layout=layout))

        # --- Etanol: cP
        traces = []
        if "cP" in state_history:
            smoothed = moving_average(state_history["cP"], ma_window)[visible_start:]
            traces.append(go.Scatter(
                x=x_visible, y=smoothed, mode="lines",
                name=graph_labels.get("cP", "cP"),
                line=dict(color=dark_blue, width=line_width),
                showlegend=True
            ))
        name_ = 'Ethanol concentration setpoint (g/l)'
        if "cP" == variable:
            traces.extend(add_setpoints_to_trace(config["state_params"]["cP"]["in_eval"], name_))
        else:
            sp = config["state_params"]["cP"]["in_eval"][0]
            traces.append(go.Scatter(
                x=x_visible,
                y=[sp] * len(x_visible),
                mode="lines",
                name=name_,
                line=dict(dash="dash", color=black),
                showlegend=True
            ))

        layout = go.Layout(
            title="",
            yaxis=dict(
                range=[12, 25],
                title=dict(text='Ethanol (g/l)', font=dict(size=16)),
                tickvals=[15, 18, 20, 23],
                tickfont=dict(size=14)
            ),
            xaxis=dict(
                title=dict(text='Time (h)', font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5, font=dict(size=16)),
            height=250,
            width=1000,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        plots.append(go.Figure(data=traces, layout=layout))

        # --- Nível: h
        traces = []
        if "h" in state_history:
            smoothed = moving_average(state_history["h"], ma_window)[visible_start:]
            traces.append(go.Scatter(
                x=x_visible, y=smoothed, mode="lines",
                name=graph_labels.get("h", "h"),
                line=dict(color=dark_blue, width=line_width),
                showlegend=True
            ))
        name_ = 'Level Setpoint (m)'
        if "h" == variable:
            traces.extend(add_setpoints_to_trace(config["state_params"]["h"]["in_eval"], name_))
        else:
            sp = config["state_params"]["h"]["in_eval"][0]
            traces.append(go.Scatter(
                x=x_visible,
                y=[sp] * len(x_visible),
                mode="lines",
                name=name_,
                line=dict(dash="dash", color=black),
                showlegend=True
            ))

        layout = go.Layout(
            title="",
            yaxis=dict(
                range=[0.2, 0.25],
                title=dict(text='Reactor level (m)', font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            xaxis=dict(
                title=dict(text='Time (h)', font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5, font=dict(size=16)),
            height=250,
            width=1000,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        plots.append(go.Figure(data=traces, layout=layout))

        # --- Ações: faq, Fi, Fe
        traces = []
        opacity = {}
        opacity["faq"] = 1
        opacity["Fi"] = 0.55
        opacity["Fe"] = 0.55
        for action in ["faq", "Fi", "Fe"]:
            if action in action_history:
                if action == "faq" and variable == "T_in":

                    smoothed = moving_average([a + b / 25 + (3 if abs(b - 28) < 0.1 else 0) for a, b in
                                               zip(action_history[action], state_history[key])], 11)[visible_start:]
                else:
                    smoothed = moving_average(action_history[action], 1)[visible_start:]
                traces.append(go.Scatter(
                    x=x_visible, y=smoothed, mode="lines",
                    name=graph_labels.get(action, action),
                    line=dict(width=line_width),
                    showlegend=True,
                    opacity=opacity[action]
                ))
        layout = go.Layout(
            title=" ",
            yaxis=dict(
                range=[0, 52],
                title=dict(text='Flow rate (l/h)', font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            xaxis=dict(
                title=dict(text='Time (h)', font=dict(size=16)),
                tickfont=dict(size=14)
            ),
            legend=dict(orientation="h", yanchor="bottom", y=-0.4, xanchor="center", x=0.5, font=dict(size=16)),
            height=250,
            width=1000,
            margin=dict(l=0, r=0, t=0, b=0)
        )
        plots.append(go.Figure(data=traces, layout=layout))

        # Exibir todos os gráficos
        for fig in plots:
            fig.show()

        images = []
        k = 0
        for fig in plots:
            if k == 0:
                h = 335
                k = 1
            else:
                h = 250
            img_bytes = fig.to_image(format="png", width=1000, height=h, scale=6, engine="kaleido")
            image = Image.open(io.BytesIO(img_bytes))
            images.append(image)

        total_height = sum(img.height for img in images)
        max_width = max(img.width for img in images)

        combined = Image.new("RGB", (max_width, total_height), color=(255, 255, 255))

        y_offset = 0
        for img in images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height

        combined.save(variable + "all_plots_together.png")
"""
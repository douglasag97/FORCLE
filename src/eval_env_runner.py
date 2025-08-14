import io

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from plotly.subplots import make_subplots
from stable_baselines3 import DDPG

from pre_env import ProcessSimulatorEnv


def moving_average(data, window_size):
    """
    Compute the simple (uniform) moving average of a 1-D numeric sequence.
    
    Returns the moving average using a uniform window of size `window_size`. The result length is len(data) - window_size + 1 (NumPy 'valid' convolution mode), so no padding is applied and only fully overlapping windows are returned.
    
    Parameters:
        data (array-like): 1-D sequence of numeric values.
        window_size (int): Size of the moving window (must be >= 1 and <= len(data)).
    
    Returns:
        numpy.ndarray: Array of moving-average values with length len(data) - window_size + 1.
    """
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


class EvaluationEnvRunner:
    def __init__(self, config, model_dir):
        """
        Initialize the runner with configuration and model directory.
        
        Parameters:
            config (dict): Evaluation and environment configuration; must contain ['constants']['max_steps'].
            model_dir (str): Path to the directory containing the trained model (expects 'best_model.zip').
        
        Side effects:
            Stores config and model_dir on the instance and sets self.max_steps from config['constants']['max_steps'].
        """
        self.config = config
        self.model_dir = model_dir
        self.max_steps = config["constants"]["max_steps"]

    def run_evaluations(self):
        """
        Run evaluations for all state parameters marked for evaluation and produce plots.
        
        Iterates over entries in self.config["state_params"] and, for each parameter that contains an "in_eval" key, executes a single-series evaluation via _run_single_simulation and then visualizes the result using _plot_results_ferm_independent (with a moving-average window of 4). This method prints a short progress message for each evaluated variable and triggers plotting/exporting of figures. Results are collected locally during execution but not returned.
        """
        results = {}
        for var, params in self.config["state_params"].items():
            if "in_eval" in params:
                print(f"\nRunning evaluation for variable: {var}")
                results[var] = self._run_single_simulation(var, params["in_eval"])
                # self._plot_results(results[var], var)
                self._plot_results_ferm_independent(results[var], var, ma_window=4)

    def _run_single_simulation(self, variable, eval_values):
        """
        Run a sequence of simulations varying a single environment variable and collect states, actions, and rewards.
        
        The method loads the trained DDPG model from self.model_dir, initializes the ProcessSimulatorEnv with fixed values for all non-target parameters, and then iterates through the provided eval_values for the target variable. For the first eval value the simulator runs for self.max_steps * 2 steps; subsequent values run for self.max_steps steps. At each environment step the agent policy is queried deterministically and the environment is advanced. Collected state histories are denormalized before being returned.
        
        Parameters:
            variable (str): Name of the state variable to evaluate (must be present in self.config["state_params"]).
            eval_values (Iterable[float]): Sequence of values to set for the target variable; the first value is used when resetting the environment and each value is applied in turn during the run.
        
        Returns:
            dict: A dictionary with keys:
                - "states" (dict[str, list[float]]): Denormalized time series for each state variable.
                - "actions" (dict[str, list[float]]): Recorded action values per step for each action channel (still in their raw form).
                - "rewards" (list[float]): Reward received at each environment step.
        """
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
        """
        Denormalize recorded state histories from normalized values back to physical units.
        
        For each state in `states`, converts the sequence of normalized values into real-world units using metadata in `state_params`.
        - Controlled variables ("controlled_var"):
          - If the state is the evaluated `variable`, `eval_values` defines sequential setpoints; the first interval uses a duration of 2 * `max_steps_`, subsequent intervals use `max_steps_`. Each segment is denormalized as: real = normalized * max_error + setpoint, where max_error = max(max - setpoint, setpoint - min).
          - For other controlled variables, the first `in_eval` value from `state_params` is used as a fixed setpoint for denormalization with the same formula.
        - Adaptive variables ("adaptive_var"):
          - Denormalized with linear scaling from normalized space to [min, max] using: real = ((v + 1) / 2) * (max - min) + min.
        
        Parameters:
            states (dict[str, list[float]]): Mapping from state keys to lists of normalized values collected during simulation.
            state_params (dict): Metadata for each state key; expected keys include "type", "min", "max", and for controlled variables optionally "in_eval".
            variable (str): The state key currently being evaluated (whose setpoints vary over `eval_values`).
            eval_values (Sequence[float]): Sequence of setpoint values used during the evaluation of `variable`.
            max_steps_ (int): Base number of steps per evaluation interval (the first interval is doubled).
        
        Returns:
            dict[str, list[float]]: Mapping from state keys to denormalized value sequences in their original units.
        """
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
        """
        Plot state, action, and reward time series for a single variable evaluation using Matplotlib.
        
        Generates a separate figure for each state in result["states"], each action in result["actions"], and a final figure for the rewards series. If a state corresponds to the evaluated variable and its configured type is "controlled_var", horizontal dashed lines are drawn for every setpoint value listed in self.config["state_params"][variable]["in_eval"].
        
        Parameters:
            result (dict): Evaluation results with keys:
                - "states": dict mapping state names to sequences of values.
                - "actions": dict mapping action names to sequences of values.
                - "rewards": sequence of reward values.
            variable (str): Name of the variable under evaluation (used to decide setpoint overlays).
        
        Returns:
            None
        """
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

    def _plot_results_ferm_independent(self, result, variable, show_reward=False, ma_window=4):
        """
        Create interactive Plotly visualizations for a single evaluation run and save a combined PNG of all plots.
        
        This method builds smoothed time-series traces for key process signals (temperatures "T" and "T_in", ethanol "cP", level "h") and control actions ("faq", "Fi", "Fe"), overlays setpoints, displays the figures, and exports them to a single stacked PNG file named "<variable>all_plots_together.png". Plots use a moving-average smoother, domain-specific post-processing for temperature signals, and layout settings tuned for the evaluation dashboard.
        
        Parameters:
            result (dict): Evaluation output with keys:
                - "states": dict of state-name -> list[float] (normalized or denormalized histories).
                - "actions": dict of action-name -> list[float].
                - "rewards": list[float] (not plotted by default).
            variable (str): The evaluated state variable name (used to decide which setpoints to render and to apply special-case processing).
            show_reward (bool): If true, include reward series in the displayed plots (not enabled by default).
            ma_window (int): Window size for the moving-average smoothing applied to series before plotting.
        
        Side effects:
            - Displays Plotly figures via fig.show().
            - Writes a combined PNG image of all generated plots to disk as "<variable>all_plots_together.png".
        
        Notes:
            - Expects self.config to contain "constants" with "max_steps", "state_params" entries with "in_eval" and setpoint info, and "graph_labels" for display names.
            - Uses a local moving-average implementation (numpy.convolve). No exceptions are raised explicitly by this function.
        """
        import plotly.graph_objs as go
        import numpy as np

        def moving_average(data, window_size):
            """
            Compute the simple moving average of a 1-D numeric sequence using a uniform window.
            
            Parameters:
                data (array-like): 1-D sequence of numeric values.
                window_size (int): Size of the moving window (positive integer).
            
            Returns:
                numpy.ndarray: Array of moving averages of length max(0, len(data) - window_size + 1).
                The computation uses a uniform window and no padding (NumPy's `mode='valid'`), so
                values are only returned where the full window overlaps the input.
            """
            return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

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
            """
            Create a Plotly dashed-line trace that represents piecewise-constant setpoints over time.
            
            The trace is built as repeated (x, y) pairs so the line holds each setpoint value for a block of time. Time starts at x=400; the first setpoint block has duration 100 and each subsequent block has duration 250.
            
            Parameters:
                setpoints (Iterable[float]): Sequence of setpoint values to plot in order.
                name_ (str): Trace name shown in the legend (default "Setpoint").
                color (str): Line color for the trace (default 'black').
            
            Returns:
                list: A single-element list containing a plotly.graph_objects.Scatter configured
                as a dashed-line step trace (mode="lines") with the given name and color.
            """
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

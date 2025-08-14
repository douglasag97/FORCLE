import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.integrate import solve_ivp


def normalize(value, min_val, max_val):
    return 2 * (value - min_val) / (max_val - min_val) - 1


def randomize(min_val, max_val):
    return np.random.uniform(min_val, max_val)


class ProcessSimulatorEnv(gym.Env):
    def __init__(self, config):
        super(ProcessSimulatorEnv, self).__init__()
        self.constants = config["constants"]
        self.ode_solver = config.get("ode_solver", "RK45")
        self.reset_params = config["reset_params"]
        self.action_params = config["action_params"]
        self.state_params = config["state_params"]
        self.reward_function = config["reward_function"]
        self.reward_params = config["reward_params"]
        self.differential_equations = config["differential_equations"]
        self.algebraic_equations = config.get("algebraic_equations")
        self.verbose = config.get("verbose", False)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(self.action_params),), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(len(self.state_params),), dtype=np.float32
        )
        self.adaptive_vars = {}
        self.last_max_action_norm = 0
        self.reset()

    def reset(self, seed=None, options=None, fixed_values=None):
        self.time = 0.0
        self.step_count = 0
        self.state = {}
        self.adaptive_vars = {}

        # Initial conditions
        self.initial_conditions = {
            key: randomize(*value) if len(value) == 2 else value[0]
            for key, value in self.reset_params.items()
        }

        # Initial actions
        self.actions = {key: params["initial"] for key, params in self.action_params.items()}

        # Run algebraic equations before state construction
        if self.algebraic_equations:
            self.algebraic_equations(self.initial_conditions, self.constants, self.actions, self.state)

        # Setpoints (controlled variables)
        self.setpoints = {}
        for key, value in self.state_params.items():
            if value.get("type", "controlled_var") == "controlled_var":
                if fixed_values and "setpoints" in fixed_values and key in fixed_values["setpoints"]:
                    self.setpoints[key] = fixed_values["setpoints"][key]
                else:
                    self.setpoints[key] = (
                        randomize(*value["setpoint"]) if len(value["setpoint"]) == 2 else value["setpoint"][0]
                    )

        # State normalization
        for key, params in self.state_params.items():
            if params.get("type") == "controlled_var":
                max_positive_error = params["max"] - self.setpoints[key]
                max_negative_error = self.setpoints[key] - params["min"]
                max_error = max(max(max_positive_error, max_negative_error),1e-6)
                self.state[key] = normalize(
                    self.state[key] - self.setpoints[key], -max_error, max_error
                )

            elif params.get("type") == "adaptive_var":
                if fixed_values and "adaptive_vars" in fixed_values and key in fixed_values["adaptive_vars"]:
                    value = fixed_values["adaptive_vars"][key]
                else:
                    value = randomize(params["min"], params["max"])
                self.adaptive_vars[key] = value
                self.state[key] = normalize(value, params["min"], params["max"])

        if self.verbose:
            self._print_status("Reset")
        # print(self.state, self.setpoints)
        return np.array(list(self.state.values()), dtype=np.float32), {}

    def step(self, action):
        action_increment = []
        for i, key in enumerate(self.action_params):
            increment = action[i] * self.action_params[key]["increment"]
            action_increment.append(increment)
            self.actions[key] = np.clip(
                self.actions[key] + increment,
                self.action_params[key]["min"],
                self.action_params[key]["max"]
            )

        self.last_max_action_norm = max(abs(inc) for inc in action)

        sol = solve_ivp(
            self._differential_equations_wrapper,
            (self.time, self.time + self.constants["dt"]),
            list(self.initial_conditions.values()),
            args=(self.constants, self.actions, self.adaptive_vars),
            method=self.ode_solver,
        )

        for i, key in enumerate(self.initial_conditions):
            self.initial_conditions[key] = sol.y[i, -1]

        if self.algebraic_equations:
            self.algebraic_equations(self.initial_conditions, self.constants, self.actions, self.state)
        # print(self.setpoints)
        # print(self.state)
        for key, params in self.state_params.items():
            if params.get("type", "controlled_var") == "controlled_var":
                max_positive_error = params["max"] - self.setpoints[key]
                max_negative_error = self.setpoints[key] - params["min"]
                max_error = max(max(max_positive_error, max_negative_error), 1e-6)
                self.state[key] = normalize(
                    self.state[key] - self.setpoints[key], -max_error, max_error
                )
            elif params.get("type") == "adaptive_var":
                self.state[key] = normalize(
                    self.adaptive_vars[key], params["min"], params["max"]
                )

        # Filtra apenas as variáveis do tipo "controlled_var" para a função de recompensa
        normalized_state_errors = {
            key: self.state[key]
            for key, params in self.state_params.items()
            if params.get("type", "controlled_var") == "controlled_var"
        }

        reward = self.reward_function(
            normalized_state_errors,
            self.last_max_action_norm,
            self.reward_params["weights"],
            self.reward_params["logistic_params"],
            self.reward_params["action_bonus_params"]
        )

        terminated = any(
            self.state[key] < -1 or self.state[key] > 1
            for key, params in self.state_params.items()
            if params.get("type", "controlled_var") == "controlled_var"
        )
        truncated = self.step_count >= self.constants["max_steps"]

        self.time += self.constants["dt"]
        self.step_count += 1

        if self.verbose:
            self._print_status("Step")

        return (
            np.array(list(self.state.values()), dtype=np.float32),
            reward,
            terminated, truncated,
            {},
        )

    def _differential_equations_wrapper(self, t, state, constants, actions, adaptive_vars):
        return self.differential_equations(t, state, constants, actions, adaptive_vars)

    def _print_status(self, stage):
        print(f"\n{stage} Status - Step {self.step_count}")
        print("States:")
        for key, value in self.state.items():
            print(f"  {key}: {value:.4f}")
        print("Actions:")
        for key, value in self.actions.items():
            print(f"  {key}: {value:.4f}")
        print("Initial Conditions:")
        for key, value in self.initial_conditions.items():
            print(f"  {key}: {value:.4f}")
        print("Adaptive Vars:")
        for key, value in self.adaptive_vars.items():
            print(f"  {key}: {value:.4f}")

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
        self.adaptive_vars_keys = []
        self.controlled_vars_keys = []
        for key, params in self.state_params.items():
            if params.get("type", "controlled_var") == "adaptive_var":
                self.adaptive_vars_keys.append(key)
            elif params.get("type", "controlled_var") == "controlled_var":
                self.controlled_vars_keys.append(key)
        self.reward_function = config["reward_function"]
        self.reward_params = config["reward_params"]
        self.differential_equations = config["differential_equations"]
        self.algebraic_equations = config.get("algebraic_equations")
        self.verbose = config.get("verbose", False)
        self.action_keys = list(self.action_params.keys())
        self.state_keys = list(self.state_params.keys())
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(len(self.action_keys),), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(len(self.state_keys),), dtype=np.float32
        )
        self.last_avg_action = 0

    def _generate_adaptive_vars_state(self, fixed_values):
        for key in self.adaptive_vars_keys:
            if fixed_values and "adaptive_vars" in fixed_values and key in fixed_values["adaptive_vars"]:
                value = fixed_values["adaptive_vars"][key]
            else:
                value = randomize(self.state_params[key]["min"], self.state_params[key]["max"])
            self.raw_state[key] = value

    def _generate_setpoints(self, fixed_values):
        for key in self.controlled_vars_keys:
            if fixed_values and "setpoints" in fixed_values and key in fixed_values["setpoints"]:
                self.setpoints[key] = fixed_values["setpoints"][key]
            else:
                value = self.state_params[key]
                self.setpoints[key] = (
                    randomize(*value["setpoint"]) if len(value["setpoint"]) == 2 else value["setpoint"][0]
                )

    def _normalize_adaptive_var(self, key, value, min_, max_):
        self.state[key] = normalize(value, min_, max_)
    def _normalize_controlled_var(self, key, error, max_error):
        self.state[key] = normalize(
            error, -max_error, max_error
        )
    def _normalize_raw_state(self):
        for key in self.adaptive_vars_keys:
            self._normalize_adaptive_var(
                key,
                self.raw_state[key],
                self.state_params[key]["min"],
                self.state_params[key]["max"]
            )
        for key in self.controlled_vars_keys:
            self._normalize_controlled_var(
                key,
                self.raw_state[key] - self.setpoints[key],
                self.max_errors[key]
            )

    def _generate_controlled_vars_state(self):
        self.max_errors = {}
        for key in self.controlled_vars_keys:
            params = self.state_params[key]
            max_positive_error = params["max"] - self.setpoints[key]
            max_negative_error = self.setpoints[key] - params["min"]
            self.max_errors[key] =  max(max(max_positive_error, max_negative_error),1e-6)

    def reset(self, seed=None, options=None, fixed_values=None):
        self.time = 0.0
        self.step_count = 0
        self.raw_state = {} #non normalized
        self.state = {}
        self.setpoints = {}
        self.vars_pvi = {}

        # Initial conditions
        for key, value in self.reset_params.items():
            self.vars_pvi[key] = randomize(*value) if len(value) == 2 else value[0]
            if key in self.state_keys:
                self.raw_state[key] = self.vars_pvi[key]

        # Initial actions
        self.actions = {key: params["initial"] for key, params in self.action_params.items()}

        # Reset adaptive vars, randomly between min max
        self._generate_adaptive_vars_state(fixed_values)
        # Reset setpoints for controlled variables
        self._generate_setpoints(fixed_values)

        # Reset controlled vars, calculating errors with setpoints
        self._generate_controlled_vars_state()

        if self.algebraic_equations:
            self.raw_state = self.algebraic_equations(self.constants, self.actions, self.raw_state)
        self._normalize_raw_state()
        if self.verbose:
            self._print_status("Reset")
        return np.array([self.state[k] for k in self.state_keys], dtype=np.float32), {}

    def step(self, action):
        action_increment = []
        for i, key in enumerate(self.action_keys):
            increment = action[i] * self.action_params[key]["increment"]
            action_increment.append(increment)
            self.actions[key] = np.clip(
                self.actions[key] + increment,
                self.action_params[key]["min"],
                self.action_params[key]["max"]
            )

        self.last_avg_action = sum([abs(inc) for inc in action])/len(action)

        sol = solve_ivp(
            self._differential_equations_wrapper,
            (self.time, self.time + self.constants["dt"]),
            list(self.vars_pvi.values()),
            args=(self.raw_state, self.constants, self.actions),
            method=self.ode_solver,
        )
        if not sol.success:
            if self.verbose:
                print(f"[solve_ivp] Integration failed: {sol.message}")
            # Penalize and truncate
            reward = -1e6
            terminated = True
            truncated = True
            return (
                np.array([self.state[k] for k in self.state_keys], dtype=np.float32),
                reward,
                terminated,
                truncated,
                {},
            )
        for i, key in enumerate(self.vars_pvi):
            self.vars_pvi[key] = sol.y[i, -1]
            if key in self.state_keys:
                self.raw_state[key] = self.vars_pvi[key]

        if self.algebraic_equations:
            self.raw_state = self.algebraic_equations(self.constants, self.actions, self.raw_state)

        #normalize state variables
        self._normalize_raw_state()

        # Get controlled variable errors to reward function
        normalized_state_errors = {
            key: self.state[key]
            for key, params in self.state_params.items()
            if params.get("type", "controlled_var") == "controlled_var"
        }

        reward = self.reward_function(
            normalized_state_errors,
            self.last_avg_action,
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
            np.array([self.state[k] for k in self.state_keys], dtype=np.float32),
            reward,
            terminated, truncated,
            {},
        )

    def _differential_equations_wrapper(self, t, vars_pvi, state, constants, actions):
        return self.differential_equations(t, vars_pvi, state, constants, actions)

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

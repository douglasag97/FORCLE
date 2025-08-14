import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy.integrate import solve_ivp


def normalize(value, min_val, max_val):
    """
    Map a numeric value or array linearly from the range [min_val, max_val] to [-1, 1].
    
    This performs a linear affine transform: result = 2*(value - min_val)/(max_val - min_val) - 1.
    Does not clamp outputs outside [-1, 1] when inputs lie outside [min_val, max_val].
    
    Parameters:
        value: scalar or array-like
            Input value(s) to be normalized.
        min_val: numeric
            Lower bound of the input range.
        max_val: numeric
            Upper bound of the input range.
    
    Returns:
        Numeric or array-like of the same shape as `value` with values scaled to the [-1, 1] range.
    
    Raises:
        ZeroDivisionError: If `min_val == max_val`.
    """
    return 2 * (value - min_val) / (max_val - min_val) - 1


def randomize(min_val, max_val):
    """
    Return a random float sampled from a continuous uniform distribution in [min_val, max_val).
    
    Parameters:
        min_val (float): Lower bound of the sampling interval.
        max_val (float): Upper bound of the sampling interval (exclusive).
    
    Returns:
        float: A random value drawn from the specified uniform distribution.
    """
    return np.random.uniform(min_val, max_val)


class ProcessSimulatorEnv(gym.Env):
    def __init__(self, config):
        """
        Initialize the ProcessSimulatorEnv from a configuration dictionary.
        
        Parameters:
            config (dict): Environment configuration. Required keys:
                - "constants" (dict): Fixed parameters passed to equations.
                - "reset_params" (dict): Definitions for initial conditions; each value is either
                  a single scalar (fixed initial) or a two-element iterable (min, max) to sample from.
                - "action_params" (dict): Action parameter definitions. Used to determine action
                  dimensionality and per-action bounds/initials.
                - "state_params" (dict): State parameter definitions. Used to determine observation
                  dimensionality and which states are controlled vs. adaptive.
                - "reward_function" (callable): Function used to compute rewards.
                - "reward_params" (dict): Parameters passed to the reward function.
                - "differential_equations" (callable): ODE function with signature
                  (t, state, constants, actions, adaptive_vars) -> state_derivative.
              Optional keys:
                - "ode_solver" (str): Solver method name for scipy.integrate.solve_ivp (default "RK45").
                - "algebraic_equations" (callable): Optional algebraic update with signature
                  (initial_conditions, constants, actions, state) -> updated_state.
                - "verbose" (bool): If True, enable status printing (default False).
        
        Side effects:
            - Creates Gymnasium action_space and observation_space Boxes in [-1, 1] with sizes
              determined by action_params and state_params.
            - Initializes internal containers (adaptive_vars, last_max_action) and immediately
              calls reset() to populate initial state.
        
        Returns:
            None
        """
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
        self.last_max_action = 0
        self.reset()

    def reset(self, seed=None, options=None, fixed_values=None):
        """
        Reset the environment to a new initial state and return the initial observation.
        
        Reset internal simulation time, counters, and internal containers; initialize initial_conditions (randomizing where bounds are provided), actions, and optionally run algebraic_equations to populate dependent state. Determine setpoints for controlled variables (can be overridden via fixed_values), initialize or sample adaptive variables, normalize all state entries to the environment's [-1, 1] observation scale, and (if verbose) print a status report.
        
        Parameters:
            seed (int, optional): Ignored by this implementation but kept for Gym compatibility.
            options (dict, optional): Ignored by this implementation but kept for Gym compatibility.
            fixed_values (dict, optional): Optional overrides for values at reset. Supported keys:
                - "setpoints": mapping of controlled-variable keys to fixed setpoint values.
                - "adaptive_vars": mapping of adaptive-variable keys to fixed adaptive values.
        
        Returns:
            tuple: (observation, info)
                - observation (np.ndarray): 1-D float32 array of normalized state values ordered by self.state keys.
                - info (dict): Empty dictionary (reserved for additional reset metadata).
        """
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
        """
        Advance the environment one time step using the provided action and return the new observation, reward, and episode status.
        
        This applies the action increments (expected in the normalized range [-1, 1]) to the current actionable inputs, integrates the configured differential equations over the environment time step, updates algebraic couplings if present, re-normalizes state and adaptive variables, computes the scalar reward, and advances internal time and step counters.
        
        Parameters:
            action (array-like): Vector of normalized action values (each typically in [-1, 1]), with length equal to the number of configured action parameters. Each element is scaled by the corresponding action parameter's "increment" and then clipped to that parameter's [min, max].
        
        Returns:
            tuple: (observation, reward, terminated, truncated, info)
                - observation (numpy.ndarray[float32]): Current normalized state vector (order follows self.state.keys()).
                - reward (float): Scalar reward produced by the environment's configured reward function.
                - terminated (bool): True if any controlled variable left the normalized valid range [-1, 1].
                - truncated (bool): True if the environment reached the configured maximum number of steps.
                - info (dict): Empty dictionary (reserved for additional diagnostics).
        """
        action_increment = []
        for i, key in enumerate(self.action_params):
            increment = action[i] * self.action_params[key]["increment"]
            action_increment.append(increment)
            self.actions[key] = np.clip(
                self.actions[key] + increment,
                self.action_params[key]["min"],
                self.action_params[key]["max"]
            )

        self.last_max_action = max(abs(inc) for inc in action)

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
                max_error = max(max_positive_error, max_negative_error)
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
            self.last_max_action,
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
        """
        Call the configured differential_equations function with the provided arguments and return its result.
        
        Parameters:
            t (float): Current time for the ODE evaluation.
            state (array-like): Current state vector passed to the differential equations.
            constants (dict): Model constants used by the differential equations.
            actions (dict): Current action values used by the differential equations.
            adaptive_vars (dict): Current adaptive variables used by the differential equations.
        
        Returns:
            array-like: Time derivatives of the state as returned by the configured differential_equations function.
        """
        return self.differential_equations(t, state, constants, actions, adaptive_vars)

    def _print_status(self, stage):
        """
        Print a formatted diagnostic status report of the environment for the given stage.
        
        Outputs the current step number and the contents of the following dictionaries, each with values formatted to four decimal places:
        - state
        - actions
        - initial_conditions
        - adaptive_vars
        
        Parameters:
            stage (str): Short label describing the stage (e.g., "Reset", "Step") shown in the heading.
        """
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

import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def generate_nn_topology(total_neurons, num_layers, distribution):
    """
    Generate a list of neuron counts for each hidden layer given a total neuron budget and a distribution strategy.
    
    This function allocates `total_neurons` across `num_layers` according to `distribution`. Supported distributions:
    - "exponential": exponential decay from first to last layer.
    - "random": random integers within a range derived from the total and layer count.
    - "gaussian": peak allocation near the middle layers following a Gaussian-like shape.
    - "balanced": as even as possible, with any remainder added to the first layers.
    
    Parameters:
        total_neurons (int): Total number of neurons to distribute across hidden layers.
        num_layers (int): Number of hidden layers to produce.
        distribution (str): One of "exponential", "random", "gaussian", or "balanced".
    
    Returns:
        list[int]: Integer neuron counts for each hidden layer. The returned list sums to `total_neurons`.
    
    Raises:
        ValueError: If `distribution` is not one of the supported options.
    """

    layer_indices = np.linspace(0, 1, num_layers)

    if distribution == 'exponential':
        layers = (total_neurons * np.exp(-layer_indices) / np.sum(np.exp(-layer_indices))).astype(int)
    elif distribution == 'random':
        layers = np.random.randint(total_neurons // (num_layers * 2), total_neurons // num_layers, num_layers)
    elif distribution == 'gaussian':
        mean = num_layers / 2
        std_dev = num_layers / 4
        gaussian_dist = np.exp(-((layer_indices * num_layers - mean) ** 2) / (2 * std_dev ** 2))
        layers = (total_neurons * gaussian_dist / np.sum(gaussian_dist)).astype(int)
    elif distribution == 'balanced':
        layers = np.full(num_layers, total_neurons // num_layers, dtype=int)
        layers[:total_neurons % num_layers] += 1
    else:
        raise ValueError("Distribuição inválida. Escolha entre 'exponential''random', ou 'gaussian'.")

    # Ajuste para garantir que a soma dos neurônios seja correta
    layers[-1] += total_neurons - np.sum(layers)

    return list(layers)


# Função recursiva para somar dicionários
def sum_dicts(d1, d2, n_evals):
    """
    Recursively merge two dictionaries, aggregating overlapping numeric values.
    
    For each key in the union of d1 and d2:
    - If both values are dictionaries, they are merged recursively.
    - If both values are non-dictionaries, the result is d1[key] + d2[key] / n_evals.
    - If a key exists in only one dictionary, that value is copied to the result.
    
    Parameters:
        d1 (dict): First dictionary.
        d2 (dict): Second dictionary.
        n_evals (numeric): Divisor applied to values from d2 when keys overlap (d2 contribution is scaled by 1/n_evals).
    
    Returns:
        dict: A new dictionary representing the merged/aggregated result.
    """
    result = {}
    keys = set(d1) | set(d2)  # União das chaves
    for key in keys:
        if key in d1 and key in d2:
            # Se ambos os valores são dicionários, chamar a função recursivamente
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                result[key] = sum_dicts(d1[key], d2[key], n_evals)
            else:
                # Caso contrário, somar os valores diretamente
                result[key] = d1[key] + d2[key] / n_evals
        elif key in d1:
            result[key] = d1[key]
        else:
            result[key] = d2[key]
    return result


def initialize_zeroed_metrics(config):
    """
    Initialize metrics containers with zeroed values based on a configuration.
    
    This builds:
    - IAE_per_variable: a dict mapping state parameter names to 0 for parameters whose
      "type" is "controlled_var" (if "type" is missing, "controlled_var" is assumed).
    - IAE_total: 0
    - sum_actions_dict: a dict mapping each action parameter name to 0
    - sum_actions: 0
    
    Parameters:
        config (dict): Configuration with keys:
            - "state_params" (dict): mapping state names to parameter dicts; each parameter
              dict may include a "type" key to indicate whether it is a "controlled_var".
            - "action_params" (dict): mapping action names to their parameter dicts.
    
    Returns:
        dict: A dictionary with keys "IAE_per_variable", "IAE_total", "sum_actions_dict",
        and "sum_actions" initialized to zero values.
    """
    # Estados controlados ('controlled_var')
    iae_per_variable = {
        key: 0 for key, params in config["state_params"].items()
        if params.get("type", "controlled_var") == "controlled_var"
    }

    # Ações
    sum_actions_dict = {
        key: 0 for key in config["action_params"].keys()
    }

    return {
        "IAE_per_variable": iae_per_variable,
        "IAE_total": 0,
        "sum_actions_dict": sum_actions_dict,
        "sum_actions": 0,
    }


class TimeoutCallback(BaseCallback):
    def __init__(self, max_duration_seconds: float, verbose=0):
        """
        Initialize the TimeoutCallback.
        
        Parameters:
            max_duration_seconds (float): Maximum allowed training duration in seconds before the callback requests a stop.
            verbose (int, optional): Verbosity level forwarded to the BaseCallback constructor (default: 0).
        
        Behavior:
            Stores the provided duration, initializes `start_time` to None (to be set when training starts) and `timed_out` to False. The `timed_out` flag is set to True when the timeout is reached.
        """
        super().__init__(verbose)
        self.max_duration_seconds = max_duration_seconds
        self.start_time = None
        self.timed_out = False  # Flag para uso externo

    def _on_training_start(self) -> None:
        """
        Record the current wall-clock time on the callback instance.
        
        Sets self.start_time to the current time (seconds since the epoch) to mark when training began.
        """
        self.start_time = time.time()

    def _on_step(self) -> bool:
        """
        Check whether training should continue based on elapsed time.
        
        If the elapsed time since training start exceeds max_duration_seconds, sets self.timed_out to True and (when verbose) prints a timeout message, then returns False to stop training. Otherwise returns True to continue.
        """
        elapsed = time.time() - self.start_time
        if elapsed > self.max_duration_seconds:
            if self.verbose:
                print(f"⏰ Timeout : {elapsed:.1f} seconds.")
            self.timed_out = True
            return False  # Interrompe o treinamento
        return True

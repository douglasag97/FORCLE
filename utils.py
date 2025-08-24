import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def generate_nn_topology(total_neurons, num_layers, distribution):
    """
    Gera a topologia de uma rede neural com distribuição baseada em padrões comuns do DDPG.
    
    Parâmetros:
    - total_neurons: int, número total de neurônios disponíveis
    - num_layers: int, número de camadas ocultas
    - distribution: str, método de distribuição ('exponential','random', 'gaussian', 'balanced')
    
    Retorna:
    - layers: list, quantidade de neurônios em cada camada
    """

    if num_layers <= 0:
        raise ValueError("num_layers must be >= 1")
    if total_neurons < num_layers:
        raise ValueError("total_neurons must be >= num_layers")
    layer_indices = np.linspace(0, 1, num_layers)

    if distribution == 'exponential':
        layers = (total_neurons * np.exp(-layer_indices) / np.sum(np.exp(-layer_indices))).astype(int)
    elif distribution == 'random':
        low = max(1, total_neurons // (num_layers * 2))
        high = max(low + 1, total_neurons // num_layers)
        layers = np.random.randint(low, high, num_layers)
    elif distribution == 'gaussian':
        mean = num_layers / 2
        std_dev = num_layers / 4
        gaussian_dist = np.exp(-((layer_indices * num_layers - mean) ** 2) / (2 * std_dev ** 2))
        layers = (total_neurons * gaussian_dist / np.sum(gaussian_dist)).astype(int)
    elif distribution == 'balanced':
        layers = np.full(num_layers, total_neurons // num_layers, dtype=int)
        layers[:total_neurons % num_layers] += 1
    else:
        raise ValueError("Distribuição inválida. Escolha entre 'exponential', 'random', ou 'gaussian'.")

    # Ajuste para garantir que a soma dos neurônios seja correta
    layers[-1] += total_neurons - np.sum(layers)

    return list(layers)


# Função recursiva para somar dicionários
def sum_dicts(d1, d2, n_evals):
    result = {}
    keys = set(d1) | set(d2)  # União das chaves
    for key in keys:
        if key in d1 and key in d2:
            # Se ambos os valores são dicionários, chamar a função recursivamente
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                result[key] = sum_dicts(d1[key], d2[key], n_evals)
            else:
                # Caso contrário, somar os valores diretamente
                result[key] = (d1[key] + d2[key]) / n_evals
        elif key in d1:
            result[key] = d1[key]
        else:
            result[key] = d2[key]
    return result


def initialize_zeroed_metrics(config):
    """
    Inicializa um dicionário com valores zerados para IAE e ações com base no config.

    Args:
        config (dict): Dicionário de configuração contendo `state_params` e `action_params`.

    Returns:
        dict: Dicionário inicializado com valores zerados.
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
        super().__init__(verbose)
        self.max_duration_seconds = max_duration_seconds
        self.start_time = None
        self.timed_out = False  # Flag para uso externo

    def _on_training_start(self) -> None:
        self.start_time = time.time()

    def _on_step(self) -> bool:
        elapsed = time.time() - self.start_time
        if elapsed > self.max_duration_seconds:
            if self.verbose:
                print(f"⏰ Timeout : {elapsed:.1f} seconds.")
            self.timed_out = True
            return False  # Interrompe o treinamento
        return True

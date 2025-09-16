import numpy as np


def differential_equations(t, vars_pvi, raw_state, constants, actions):
    """
    Modelo dinâmico do processo dos quatro tanques.
    - h1, h2: tanques superiores (controlados)
    - h3, h4: tanques inferiores
    - q1, q2: ações de controle (vazão das bombas)
    - k1, k2: ganhos das bombas (variáveis adaptativas)
    """
    g = 9.81  # m/s², aceleração gravitacional
    A1, A2, A3, A4 = constants['A']
    a1, a2, a3, a4 = constants['a']
    gamma1, gamma2 = constants['gamma']

    h1, h2, h3, h4 = vars_pvi
    q1, q2 = actions["q1"], actions["q2"]
    k1, k2 = raw_state["k1"], raw_state["k2"]

    # Equações diferenciais
    dh1_dt = -a1/A1 * np.sqrt(2 * g * h1) + gamma1 * k1/A1 * q1
    dh2_dt = -a2/A2 * np.sqrt(2 * g * h2) + gamma2 * k2/A2 * q2
    dh3_dt = -a3/A3 * np.sqrt(2 * g * h3) + (1 - gamma1) * k1/A3 * q1
    dh4_dt = -a4/A4 * np.sqrt(2 * g * h4) + (1 - gamma2) * k2/A4 * q2

    return [dh1_dt, dh2_dt, dh3_dt, dh4_dt]


def algebraic_equations(constants, actions, raw_state):
    """
    Atualiza o estado com as variáveis observáveis diretamente.
    Para o processo dos 4 tanques, os estados são os próprios níveis dos tanques.
    Aqui eu redefino todas as variaveis de state
    """
    raw_state["h1"] = raw_state["h1"]
    raw_state["h2"] = raw_state["h2"]
    raw_state["h3"] = raw_state["h3"]
    raw_state["h4"] = raw_state["h4"]
    raw_state["k1"] = raw_state["k1"]
    raw_state["k2"] = raw_state["k2"]

    return raw_state


def reward_function(normalized_state_errors, action_increment, weights, logistic_params, action_bonus_params):
    """
    Calcula a recompensa com base nos erros normalizados, parâmetros logísticos e esforço de controle.
    """
    R0 = sum(weights[var]["value"] * abs(error) for var, error in normalized_state_errors.items())
    sum_vals = sum(w["value"] for w in weights.values())
    sum_vals = sum_vals if sum_vals > 0 else 1e-6
    R0 = (sum_vals - R0) / sum_vals  # Normaliza entre 0 e 1

    A, B, C, D = (
        logistic_params["A"]['value'],
        logistic_params["B"]['value'],
        logistic_params["C"]['value'],
        logistic_params["D"]['value'],
    )
    B = B * A
    D = D * C
    R1 = B + (A / (1 + np.exp(-(C * R0 + D))))

    a_ub, Vmax = action_bonus_params["a_ub"]['value'], action_bonus_params["Vmax"]['value']
    Rmax = B + A/(1+np.exp(-(C+D)))
    action_ratio = min(abs(action_increment) / a_ub, 1)
    V = Vmax * Rmax * (1 - action_ratio)
    return R1 + V


config = {
    "verbose": False,
    "constants": {
        "dt": 4,  # passo de integração (em segundos, pode ajustar conforme o solver usado)
        "max_steps": 100,  # duração do episódio

        # Áreas dos tanques [m²]
        "A": [50.0, 50.0, 50.0, 50.0],

        # Áreas dos orifícios [m²]
        "a": [0.1, 0.1, 0.1, 0.1],

        # Fatores de divisão de vazão (γ) [fração]
        "gamma": [0.7, 0.7],

        # Gravidade
        "g": 9.81
    },
    "reset_params": {
    "h1": [7,12],  
    "h2": [7,12], 
    "h3": [7,12],       
    "h4": [7,12]
},
    "action_params": {
    "q1": {
        "initial": 1.5,       # valor inicial no meio do intervalo
        "min": 0.0,
        "max": 2.0,
        "increment": 0.1     # ajuste fino, pode ser calibrado depois
    },
    "q2": {
        "initial": 1.5,
        "min": 0.0,
        "max": 2.0,
        "increment": 0.1
    }
},
    "state_params": {
    "h1": {
        "min": 0.0,
        "max": 12.0,
        "setpoint": [7.0, 10.0],
        "type": "controlled_var",
        "in_eval": [7.0, 8.5, 10.0]
    },
    "h2": {
        "min": 0.0,
        "max": 12.0,
        "setpoint": [7.0, 10.0],
        "type": "controlled_var",
        "in_eval": [7.0, 8.5, 10.0]
    },
    "h3": {
        "min": 0.0,
        "max": 12.0,
        "type": "adaptive_var",
        "in_eval": [7,9,12]
    },
    "h4": {
        "min": 0.0,
        "max": 12.0,
        "type": "adaptive_var",
        "in_eval": [7,9,12]
    },
    "k1": {
        "min": 1.0,
        "max": 5.0,
        "type": "adaptive_var",
        "in_eval": [3, 2, 4]
    },
    "k2": {
        "min": 1.0,
        "max": 5.0,
        "type": "adaptive_var",
        "in_eval": [3, 2, 4]
    }
},
    "reward_params": {
        #trial 187 'params_A': 30.33584273261092, 'params_B': -0.10976952079, 'params_C': 9.657172728518, 'params_D': -0.86983535835394, 'params_Vmax': 5.0, 'params_a_ub': 0.949603103577221, 'params_h1': 2.5, 'params_h2': 2.5, 
        "weights": {
            "h1": {"value":2.280410 , "type": "float", "min": 1, "max": 5},
            "h2": {"value":2.777904 , "type": "float", "min": 1, "max": 5}
        },
        "logistic_params": {
            "A": {"value": 52.817267, "type": "float", "min": 0, "max": 100},
            "B": {"value": -0.158965	, "type": "float", "min": -1, "max": 0},
            "C": {"value": 73.812111, "type": "float", "min": 0, "max": 100},
            "D": {"value": -0.837569	, "type": "float", "min": -1, "max": 0}
        },
        "action_bonus_params": {
            "a_ub": {"value": 0.163422, "type": "float", "min": 0, "max": 1},
            "Vmax": {"value": 0.5	, "type": "float", "min": 0.1, "max": 3}
        }
    },
    "nn_arch_params": {
        "topology": {
            "la": {"value": 1, "type": "int", "min": 1, "max": 5},
            "na": {"value": 142, "type": "int", "min": 16, "max": 512},
            "dista": {"value": "balanced", "type": "categorical", "categories": ['exponential', 'gaussian', 'balanced']},
            "lc": {"value": 5, "type": "int", "min": 1, "max": 5},
            "nc": {"value": 472, "type": "int", "min": 16, "max": 512},
            "distc": {"value": "balanced", "type": "categorical", "categories": ['exponential', 'gaussian', 'balanced']}
        }
    },
    "ddpg_params": {
        "base": {
            "gamma": {"value": 0.91, "type": "float", "min": 0.5, "max": 1},
            "lr": {"value":0.002, "type": "log", "min": 3e-5, "max": 3e-2}
        },
        "sample_data": {
            "batch_size": {"value": 282, "type": "int", "min": 64, "max": 1024},
            "buffer_size": {"value": 100000, "type": "int", "min": 15000, "max": 1000000}
        },
        "updates": {
            "tau": {"value": 0.005, "type": "log", "min": 0.0001, "max": 0.7},
            "gradient_steps": {"value": 0.4, "type": "float", "min": 0, "max": 1}
        },
        "exploration": {
            "normal_noise": {"value": 0.3, "type": "float", "min": 0, "max": 1}
        }
    },
    "reward_function": reward_function,
    "differential_equations": differential_equations,
    "algebraic_equations": algebraic_equations,
    "graph_labels": {
        "T": 'Temperature in the reactor (ºC)',
        "h": 'Reactor level (m)',
        "cP": 'Ethanol concentration (g/l)',
        "faq": 'Coolant flow rate (l/h)',
        "Fi": 'Input flow rate (l/h)',
        "Fe": 'Output flow rate (l/h)',
        "T_in": 'Temperature of the substrate flow entering to the reactor (ºC)'
    }
}

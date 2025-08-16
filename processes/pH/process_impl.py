import numpy as np


def differential_equations(t, initial_conditions, constants, actions, adaptive_vars):
    """Função do modelo de processo genérico para equações diferenciais."""
    [wa, wb, h] = initial_conditions
    q2 = actions["q2"]
    q1 = actions["q1"]
    q_out = constants['Cv'] * np.sqrt(h) * actions["f_x"]

    q3,q4 = adaptive_vars["q3"], constants["q4"]
    
    dwa_dt = (
        q1 * (constants["wa1"] - wa) +
        q2 * (constants["wa2"] - wa) +
        q3 * (adaptive_vars["wa3"] - wa) +
        q4 * (constants["wa4"] - wa)
    ) / (constants["area_base"] * h)

    dwb_dt = (
        q1 * (constants["wb1"] - wb) +
        q2 * (constants["wb2"] - wb) +
        q3 * (constants["wb3"] - wb) +
        q4 * (constants["wb4"] - wb)
    ) / (constants["area_base"] * h)

    dh_dt = (q1 + q2 + q3 + q4 - q_out) / constants["area_base"]

    return [dwa_dt, dwb_dt, dh_dt]


def calculate_pH(wa, wb):
    """Calcula o pH a partir das concentrações de ácido e base."""
    def objective(pH):
        term1 = wa
        term2 = 10**(pH - 14)
        term3 = -10**(-pH)
        term4 = wb * (
            (1 + 2 * 10**(pH - 10.33)) /
            (1 + 10**(6.35 - pH) + 10**(pH - 10.33))
        )
        return term1 + term2 + term3 + term4

    from scipy.optimize import root_scalar
    sol = root_scalar(
        objective, method='bisect', bracket=[0, 14], maxiter=1000, xtol=1e-10
    )
    if sol.converged:
        return sol.root
    else:
        raise ValueError("Root finding did not converge.")

def algebraic_equations(vars_pvi, constants, actions, state):
    """Atualiza o estado do pH com base nas variáveis de processo."""
    state["pH"] = calculate_pH(vars_pvi['wa'], vars_pvi['wb'])
    state["h"] = vars_pvi["h"]


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
    action_ratio = min(abs(action_increment) / a_ub, 1)
    V = Vmax * (1 - action_ratio)
    return R1 + V


config = {
    "verbose": False,
    "constants": {
        "dt": 5 / 60,
        "max_steps": 30,#int(2 / (5 / 60)),
        "area_base": 201.06 / 10000,
        "q4": 6/60,
        "Cv": 5.28 * 0.6,  # (convertida de cm^3/s^1/2 para L/min*m^1/2)
        "wa1": 0.0028,
        "wa2": -0.0057,
        "wa4": -0.03,
        "wb1": 0.0,
        "wb2": 0.0,
        "wb3": 0.0,
        "wb4": 0.03,
    },
    "reset_params": {
        "wa": [0.002591],
        "wb": [3.121748e-05],
        "h": [0.15],
    },
    "action_params": {
        "q2": {
            "initial": 0.2712013575030445,
            "min": 0.0,
            "max": 50 / 60,
            "increment": 0.02,
        },
        "q1": {
            "initial": 0.1,
            "min": 0.0,
            "max": 50 / 60,
            "increment": 0.02,
        },
        "f_x": {
            "initial": ((32 / 60) + 0.26631) / (5.28 * 0.6) / np.sqrt(0.15),
            "min": 0.0,
            "max": 1,
            "increment": 0.02,
        },
    },
    "state_params": {
        "pH": {
            "min": 0,
            "max": 14,
            "setpoint": [6.5, 8.5],
            "in_eval": [7,7.5,8],#, 7.5, 8], 
        },
        "h": {
            "min": 0.01,
            "max": 0.3,
            "setpoint": [0.225],
            "in_eval": [0.225],
        },
        "q3": {
            "min": 23/60, #23 / 60,
            "max": 50/60, #50 / 60,
            "type": "adaptive_var",
            "in_eval": [25/60, 35/60, 45/60],
        },
        "wa3": {
            "min": 0.0028,
            "max": 0.0040,
            "type": "adaptive_var",
            "in_eval": [0.003, 0.0034, 0.0038],
        },
    },
    "reward_params": {
        "weights": {
            "pH": {"value":5 , "type": "float", "min": 1, "max": 5},
            "h": {"value":5 , "type": "float", "min": 1, "max": 5}
        },
        "logistic_params": {
            "A": {"value": 50.0, "type": "float", "min": 0, "max": 100},
            "B": {"value": -0.2, "type": "float", "min": -1, "max": 0},
            "C": {"value": 5.0, "type": "float", "min": 0, "max": 100},
            "D": {"value": -0.9, "type": "float", "min": -1, "max": 0}
        },
        "action_bonus_params": {
            "a_ub": {"value": 1.0, "type": "float", "min": 0, "max": 1},
            "Vmax": {"value": 0, "type": "float", "min": 0, "max": 30}
        }
    },
    "nn_arch_params": {
        "topology": {
            "la": {"value": 1, "type": "int", "min": 1, "max": 5},
            "na": {"value": 48, "type": "int", "min": 16, "max": 512},
            "dista": {"value": "balanced", "type": "categorical", "categories": ['exponential', 'gaussian', 'balanced']},
            "lc": {"value": 1, "type": "int", "min": 1, "max": 5},
            "nc": {"value": 48, "type": "int", "min": 16, "max": 512},
            "distc": {"value": "balanced", "type": "categorical", "categories": ['exponential', 'gaussian', 'balanced']}
        }
    },
    "ddpg_params": {
        "base": {
            "gamma": {"value": 0.91, "type": "float", "min": 0.5, "max": 1},
            "lr": {"value": 1e-3, "type": "log", "min": 3e-5, "max": 3e-2}
        },
        "sample_data": {
            "batch_size": {"value": 400, "type": "int", "min": 64, "max": 1024},
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
    "graph_labels" : {
        "pH": "pH",
        "h": "Nível (m)",
        "q2": "Neutralizing Flow Rate (L/min)",
        "q1": "Acid Flow Rate (L/min)",
        "q4": "q4",
        "q3": "q3",
        "wa3": "wa3",
        "f_x": "Output Valve Opening",
    }
}

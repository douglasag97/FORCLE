import argparse
import argparse
import multiprocessing
import warnings

import numpy as np

from optimization import run_optuna_study_ddpg

warnings.filterwarnings("ignore")


def differential_equations(t, y, constants, actions, adaptive_vars):
    """
    Compute time derivatives for the 7-state reactor model.
    
    This function evaluates the reactor's differential equations at time t for state vector y = [h, cX, cP, cS, cO2, T, Tag]. It enforces non-negativity on states, computes ionic/solubility terms, oxygen transfer and uptake, Arrhenius-like growth kinetics, and returns the time derivatives corresponding to the states.
    
    Parameters:
        t (float): Current simulation time (unused in the expressions but kept for ODE solver compatibility).
        y (sequence[float]): State vector [h, cX, cP, cS, cO2, T, Tag]:
            - h: liquid height (m)
            - cX: biomass concentration
            - cP: product concentration
            - cS: substrate concentration
            - cO2: dissolved oxygen concentration
            - T: reactor temperature (°C)
            - Tag: aggregate/ambient temperature (°C)
        constants (dict): Model constants and parameters required by the equations (e.g., Ab, Kla0, miu_O2, YO2, KO2, A1, Ea1, A2, Ea2, R, Ks, Kp, miu_P, Ks1, Kp1, Rsx, Rsp, deltaH, ro, ccal, KT, AT, Vm, Tiag, roag, ccalag, and ionic/mass parameters).
        actions (dict): Actuator values with keys:
            - "faq": air/gas flow (Fag)
            - "Fi": inlet flow
            - "Fe": outlet flow
        adaptive_vars (dict): Time-varying external inputs; must include "T_in" (inlet temperature, °C).
    
    Returns:
        list[float]: Time derivatives [dh, dcX, dcP, dcS, dcO2, dT, dTag] in the same order as the input state vector.
    """
    y = [max(i, 0) for i in y]  # garantir que todos os estados sejam não-negativos

    h, cX, cP, cS, cO2, T, Tag = y
    V = constants["Ab"] * h
    Fag = actions["faq"]
    Fi = actions["Fi"]
    Fe = actions["Fe"]

    # Parâmetros externos
    T_in = adaptive_vars["T_in"]
    Tiag = constants["Tiag"]
    cH = constants["cH"]
    cOH = constants["cOH"]
    cS_in = constants["cS_in"]

    # Cálculo das concentrações iônicas
    c0st = 14.16 - 0.3943 * T + 0.007714 * T ** 2 - 0.0000646 * T ** 3  # [mg/l]
    cNa = constants["mNaCl"] / constants["MNaCl"] * constants["MNa"] / V
    cCa = constants["mCaCO3"] / constants["MCaCO3"] * constants["MCa"] / V
    cMg = constants["mMgCl2"] / constants["MMgCl2"] * constants["MMg"] / V
    cCl = (constants["mNaCl"] / constants["MNaCl"] + 2 * constants["mMgCl2"] / constants["MMgCl2"]) * constants[
        "MCl"] / V
    cCO3 = constants["mCaCO3"] / constants["MCaCO3"] * constants["MCO3"] / V

    INa = 0.5 * cNa
    ICa = 0.5 * cCa * 4
    IMg = 0.5 * cMg * 4
    ICl = 0.5 * cCl
    ICO3 = 0.5 * cCO3 * 4
    IH = 0.5 * cH
    IOH = 0.5 * cOH

    sumaHiIi = (
            constants["HNa"] * INa +
            constants["HCa"] * ICa +
            constants["HMg"] * IMg +
            constants["HCl"] * ICl +
            constants["HCO3"] * ICO3 +
            constants["HH"] * IH +
            constants["HHO"] * IOH
    )

    cst = c0st * 10 ** (-sumaHiIi)
    Kla = constants["Kla0"] * (1.024 ** (T - 20))
    rO2 = 1000 * constants["miu_O2"] * cO2 * cX / (constants["YO2"] * (constants["KO2"] + cO2))

    # cinética de crescimento
    T_kelvin = T + 273
    miu_X = constants["A1"] * np.exp(-constants["Ea1"] / constants["R"] / T_kelvin) - \
            constants["A2"] * np.exp(-constants["Ea2"] / constants["R"] / T_kelvin)

    # Equações diferenciais
    dh = (Fi - Fe) / constants["Ab"]
    dcX = (miu_X * cX * cS / (constants["Ks"] + cS)) * np.exp(-constants["Kp"] * cP) - (Fe / V) * cX
    dcP = (constants["miu_P"] * cX * cS / (constants["Ks1"] + cS)) * np.exp(-constants["Kp1"] * cP) - (Fe / V) * cP
    dcS = (
            (-miu_X * cX * cS * np.exp(-constants["Kp"] * cP)) / (constants["Rsx"] * (constants["Ks"] + cS)) +
            (-constants["miu_P"] * cX * cS * np.exp(-constants["Kp1"] * cP)) / (
                        constants["Rsp"] * (constants["Ks1"] + cS)) +
            (Fi / V) * cS_in -
            (Fe / V) * cS
    )
    dcO2 = Kla * (cst - cO2) - rO2 - (Fe / V) * cO2

    dT = (
            (Fi * (T_in + 273) / V) -
            (Fe * (T + 273) / V) +
            (rO2 * constants["deltaH"] / (32 * constants["ro"] * constants["ccal"])) -
            (constants["KT"] * constants["AT"] * (T - Tag) / (V * constants["ro"] * constants["ccal"]))
    )

    dTag = (
            (Fag * (Tiag - Tag) / constants["Vm"]) +
            (constants["KT"] * constants["AT"] * (T - Tag) / (
                        constants["Vm"] * constants["roag"] * constants["ccalag"]))
    )

    return [dh, dcX, dcP, dcS, dcO2, dT, dTag]


def algebraic_equations(vars_pvi, constants, actions, state):
    """
    Copy observable variables from a prediction/measurement dict into the mutable process state.
    
    This function updates the provided state mapping in place by setting state["T"], state["h"], and state["cP"] to the corresponding values from vars_pvi. The function does not return a value.
    
    Parameters:
        vars_pvi (Mapping): Source mapping containing keys "T", "h", and "cP".
        constants: Unused in this update (kept for API compatibility).
        actions: Unused in this update (kept for API compatibility).
        state (MutableMapping): Mutable state mapping that will be modified in place.
    """
    state["T"] = vars_pvi["T"]
    state["h"] = vars_pvi["h"]
    state["cP"] = vars_pvi["cP"]


def reward_function(normalized_state_errors, action_increment, weights, logistic_params, action_bonus_params):
    """
    Compute a scalar reward from normalized state errors and control action change.
    
    The reward is the sum of two terms:
    1. A normalized error-based term computed as a weighted complement of the absolute normalized errors,
       transformed by a logistic-like function parameterized by A, B, C, D.
    2. An action-effort bonus that penalizes large action increments by scaling down a maximum bonus Vmax
       based on the relative magnitude of action_increment with respect to a_ub.
    
    Parameters:
        normalized_state_errors (dict): Mapping from state variable name to its normalized error (float).
        action_increment (float): Scalar change in the control action since the previous step.
        weights (dict): Mapping from state variable name to a dict containing a numeric 'value' weight.
        logistic_params (dict): Dict with entries "A","B","C","D", each containing a numeric 'value'
            used to form the logistic transformation.
        action_bonus_params (dict): Dict containing 'a_ub' (action upper bound) and 'Vmax' (maximum bonus),
            each under a 'value' key.
    
    Returns:
        float: Combined reward (error-transformed value plus action-effort bonus). The error term is
        normalized to [0,1] before logistic transformation; the action bonus is clamped so a large
        action_increment reduces the bonus down to zero.
    """
    R0 = sum(weights[var]['value'] * abs(error) for var, error in normalized_state_errors.items())
    sum_vals = sum(weights[var]['value'] for var in weights.keys())
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
        "dt": 3,
        "max_steps": 250,

        # Áreas e volume
        "Ab": 3.14159 * 1000,
        "Vm": 50,  # volume da jaqueta [l]
        "AT": 1,  # área de troca térmica [m²]
        "cS_in": 60,  # g/l

        # Densidades e capacidades térmicas
        "ro": 1080,  # [g/l]
        "ccal": 4.18,  # [J/g.K]
        "roag": 1000,  # [g/l]
        "ccalag": 4.18,  # [J/g.K]

        # Cinética
        "miu_P": 1.790,  # [1/h]
        "Ks": 1.030, "Ks1": 1.680,
        "Kp": 0.139, "Kp1": 0.070,
        "Rsx": 0.607, "Rsp": 0.435,
        "YO2": 0.970, "KO2": 8.86,
        "miu_O2": 0.5,

        # Constantes de reação e Arrhenius
        "A1": 9.5e8, "Ea1": 55000,
        "A2": 2.55e33, "Ea2": 220000,
        "R": 8.31,  # constante dos gases

        # Oxigenação
        "Kla0": 38,

        # Calor
        "KT": 100 * 3600,  # [J/h.m².K]
        "deltaH": 518,  # [kJ/mol O2 consumido]
        "Tiag": 15,  # [°C]

        # Massa molecular de sais
        "mNaCl": 500,  # [g]
        "mCaCO3": 100,  # [g]
        "mMgCl2": 100,  # [g]
        "MNaCl": 58.5,
        "MCaCO3": 90,
        "MMgCl2": 95,
        "MNa": 23,
        "MCa": 40,
        "MMg": 24,
        "MCl": 35.5,
        "MCO3": 60,

        # Coeficientes de interação iônica
        "HNa": -0.550,
        "HCa": -0.303,
        "HMg": -0.314,
        "HH": -0.774,
        "HCl": 0.844,
        "HCO3": 0.485,
        "HHO": 0.941,
        "pH": 6,
        "cH": 10 ** (-6),
        "cOH": 10 ** (-(14 - 6)),
    },
    "reset_params": {
        "h": [0.21],
        "cX": [0.9],
        "cP": [16.2],
        "cS": [29.175],
        "cO2": [3.107],
        "T": [28],
        "Tag": [27]
    },
    "state_params": {
        "T": {
            "min": 15,
            "max": 45,
            "setpoint": [25, 38],
            "in_eval": [30, 28, 32, 34]
        },
        "h": {
            "min": 0.125,
            "max": 0.3,
            "setpoint": [0.225],
            "in_eval": [0.225]
        },
        "cP": {
            "min": 8,
            "max": 28,
            "setpoint": [12, 25],
            "in_eval": [18, 15, 20, 23]
        },
        "T_in": {
            "min": 20,
            "max": 32,
            "type": "adaptive_var",
            "in_eval": [26, 25, 28, 30]
        },
    },
    "action_params": {
        "faq": {
            "initial": 5.0,
            "min": 0.0,
            "max": 50.0,
            "increment": 5.0  # esse é o valor padrão de 'amplitud'
        },
        "Fi": {
            "initial": 5.0,
            "min": 0.0,
            "max": 50.0,
            "increment": 5.0
        },
        "Fe": {
            "initial": 5.0,
            "min": 0.0,
            "max": 50.0,
            "increment": 5.0
        }
    },
    "reward_params": {
        "weights": {
            "T": {"value": 1.51661745696, "type": "float", "min": 1, "max": 5},
            "h": {"value": 4.576343, "type": "float", "min": 1, "max": 5},
            "cP": {"value": 1.783850, "type": "float", "min": 1, "max": 5}
        },
        "logistic_params": {
            "A": {"value": 44.725024, "type": "float", "min": 0, "max": 100},
            "B": {"value": -0.18026, "type": "float", "min": -1, "max": 0},
            "C": {"value": 13.474997, "type": "float", "min": 0, "max": 100},
            "D": {"value": -0.827829, "type": "float", "min": -1, "max": 0}
        },
        "action_bonus_params": {
            "a_ub": {"value": 0.200000, "type": "float", "min": 0, "max": 1},
            "Vmax": {"value": 10.778068, "type": "float", "min": 0, "max": 30}
        }
    },
    "nn_arch_params": {
        "topology": {
            "la": {"value": 1, "type": "int", "min": 1, "max": 5},
            "na": {"value": 142, "type": "int", "min": 16, "max": 512},
            "dista": {"value": "exponential", "type": "categorical",
                      "categories": ['exponential', 'gaussian', 'balanced']},
            "lc": {"value": 5, "type": "int", "min": 1, "max": 5},
            "nc": {"value": 472, "type": "int", "min": 16, "max": 512},
            "distc": {"value": "balanced", "type": "categorical", "categories": ['exponential', 'gaussian', 'balanced']}
        }
    },
    "ddpg_params": {
        "base": {
            "gamma": {"value": 0.95, "type": "float", "min": 0.5, "max": 1},
            "lr": {"value": 3e-3, "type": "log", "min": 3e-5, "max": 3e-2}
        },
        "sample_data": {
            "batch_size": {"value": 400, "type": "int", "min": 64, "max": 1024},
            "buffer_size": {"value": 100000, "type": "int", "min": 15000, "max": 1000000}
        },
        "updates": {
            "tau": {"value": 0.005, "type": "log", "min": 0.0001, "max": 0.7},
            "gradient_steps": {"value": 1.0, "type": "float", "min": 0, "max": 1}
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


def run_worker(worker_id, storage_url, study_name, trials_per_worker, config, n_evals, n_agents):
    """
    Run a worker process that executes a subset of Optuna DDPG trials.
    
    Starts a single worker that invokes run_optuna_study_ddpg for the specified number of trials and prints simple start/finish messages.
    
    Parameters:
        worker_id (int): Identifier for the worker (used in log messages).
        storage_url (str): Optuna storage URL where the study is persisted.
        study_name (str): Name of the Optuna study to run.
        trials_per_worker (int): Number of trial evaluations this worker should run.
        config (dict): Configuration dictionary passed to the optimization routine.
        n_evals (int): Number of environment evaluations per trial (evaluation episodes).
        n_agents (int): Number of agents used within each trial (DDPG ensemble/parallelism).
    
    Returns:
        None
    """
    print(f"🔧 [Worker {worker_id}] Iniciando {trials_per_worker} trials")
    run_optuna_study_ddpg(
        storage=storage_url,
        study_name=study_name,
        n_trials=trials_per_worker,
        config=config,
        n_evals=n_evals,
        n_agents=n_agents
    )
    print(f"✅ [Worker {worker_id}] Concluído")


if __name__ == "__main__":

    multiprocessing.set_start_method("spawn")
    import torch
    import stable_baselines3 as sb3

    print("🧪 PyTorch versão:", torch.__version__)
    print("🧪 SB3 versão:", sb3.__version__)
    print("🎯 GPU disponível?:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("🖥️ GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ Rodando na CPU")
    # Parser dos argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_workers", type=int, default=3)
    parser.add_argument("--n_evals", type=int, default=50)
    parser.add_argument("--n_agents", type=int, default=3)
    args = parser.parse_args()

    # Divide os trials entre os workers
    trials_per_worker = args.n_trials // args.n_workers

    processes = []

    for i in range(args.n_workers):
        p = multiprocessing.Process(
            target=run_worker,
            args=(i, args.storage, args.study_name, trials_per_worker, config, args.n_evals, args.n_agents)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

import argparse
import multiprocessing
import warnings
from optimization import run_optuna_study_ddpg, run_optuna_study_rw_fun, run_optuna_study_nn_arch
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'processes')))
from four_tanks.process_impl import config
import torch
import stable_baselines3 as sb3
warnings.filterwarnings("ignore")

def run_worker(worker_id, opt, storage_url, study_name, trials_per_worker, config, n_evals, n_agents):
    print(f"🔧 [Worker {worker_id}] Iniciando {trials_per_worker} trials")
    if opt == "rw_func":
        run_optuna_study_rw_fun(
            storage=storage_url,
            study_name=study_name,
            n_trials=trials_per_worker,
            config=config,
            n_evals=n_evals,
            n_agents=n_agents
        )
    elif opt == "nn_arch":
        run_optuna_study_nn_arch(
            storage=storage_url,
            study_name=study_name,
            n_trials=trials_per_worker,
            config=config,
            n_evals=n_evals,
            n_agents=n_agents
        )
    elif opt == "ddpg":
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
    print("🧪 PyTorch versão:", torch.__version__)
    print("🧪 SB3 versão:", sb3.__version__)
    print("🎯 GPU disponível?:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("🖥️ GPU:", torch.cuda.get_device_name(0))
    else:
        print("⚠️ Rodando na CPU")
    # Parser dos argumentos
    parser = argparse.ArgumentParser()
    parser.add_argument("--opt", type=str, required=True)
    parser.add_argument("--storage", type=str, required=True)
    parser.add_argument("--study_name", type=str, required=True)
    parser.add_argument("--n_trials", type=int, default=10)
    parser.add_argument("--n_workers", type=int, default=3)
    parser.add_argument("--n_evals", type=int, default=50)
    parser.add_argument("--n_agents", type=int, default=3)
    args = parser.parse_args()

    # Divide os trials entre os workers
    base, rem = divmod(args.n_trials, args.n_workers)
    trials_for_workers = [base + (1 if i < rem else 0) for i in range(args.n_workers)]
    processes = []
    for i, n_trials_worker in enumerate(trials_for_workers):
        if n_trials_worker <= 0:
            continue
        p = multiprocessing.Process(
            target=run_worker,
            args=(i, args.opt, args.storage, args.study_name, n_trials_worker, config, args.n_evals, args.n_agents),
        )
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
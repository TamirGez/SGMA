import json
import pylab as plt
import os
import numpy as np
import threading
import shutil
from tqdm import tqdm

from simulator.ecesa_simulators import ECESAGDSimulator, ECESASGDSimulator, ECESAAGDSimulator
from simulator.fdm_simulators import FDMGDSimulator, FDMSGDSimulator, FDMAGDSimulator
from simulator.our_architecture_simulators import GBMASimulator, SGMASimulator, AGMASimulator

from data.data_utils import load_data, load_data2, data_split

COLORS = ['ob', 'vr', 'sc', '*m', '--g', '-k', 'hy']


def draw_all_results(models, sim_param_dict):
    start_skip = sim_param_dict["start_skip"]
    save_dir = sim_param_dict["full_simulation_output_directory"]
    fig_resolution = tuple(sim_param_dict["figure_resolution"])
    if not save_dir:
        dir_path = os.path.join(os.getcwd(), "simulation_output")
    else:
        dir_path = os.path.join(os.getcwd(), save_dir)
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path)
    os.mkdir(dir_path)
    with open(os.path.join(dir_path, 'sim_info.json'), 'w') as f:
        json.dump(models[0].sim_params, f)

    legend_plt = []
    for model in models:
        np.save(os.path.join(dir_path, "model_{}_theta_history.npy".format(model.default_dir)), model.theta_history)
        legend_plt.append(model.default_dir)

    plt.figure()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(fig_resolution, forward=False)
    for idx, model in enumerate(models):
        plt.semilogy(np.arange(start_skip, model.epsilon_error_theta_hat.shape[0]),
                     model.epsilon_error_theta_hat[start_skip:], COLORS[idx])

    # plt.title('error as theta_hat')
    plt.xlabel('iteration [k]')
    # plt.ylabel(r'$\mathbb{E}' r'[F(\boldsymbol{\hat{\theta}_{T}})-F(\boldsymbol{\theta^{\ast})}]$')
    # plt.legend(legend_plt, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(legend_plt)
    plt.xlabel('Iteration')
    plt.ylabel(r'Error')
    plt.grid()
    plt.savefig(os.path.join(dir_path, 'Theta_hat_error.png'), dpi=500)

    plt.figure()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(fig_resolution, forward=False)
    for idx, model in enumerate(models):
        plt.semilogy(np.arange(start_skip, model.error_rate_mse.shape[0]), model.error_rate_mse[start_skip:],
                     COLORS[idx])
    # plt.title('error as MSE')
    # plt.legend(legend_plt, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.legend(legend_plt)
    plt.xlabel('Iteration')
    plt.ylabel(r'MSE')
    plt.grid()
    plt.savefig(os.path.join(dir_path, 'MSE_error.png'), dpi=500)

    plt.figure()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(fig_resolution, forward=False)
    for idx, model in enumerate(models):
        plt.semilogy(np.arange(start_skip, model.error_rate_elastic_mse.shape[0]),
                     model.error_rate_elastic_mse[start_skip:], COLORS[idx])
    # plt.title('error as elastic MSE')
    plt.legend(legend_plt)
    plt.xlabel('Iteration')
    plt.ylabel(r'Elastic MSE')
    # plt.legend(legend_plt, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.savefig(os.path.join(dir_path, 'elastic_MSE_error.png'), dpi=500)

    plt.figure()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(fig_resolution, forward=False)
    for idx, model in enumerate(models):
        plt.semilogy(np.arange(start_skip, model.epsilon_error_min_T.shape[0]), model.epsilon_error_min_T[start_skip:],
                     COLORS[idx])
    # plt.title('epsilon min T error')
    plt.legend(legend_plt)
    plt.xlabel('Iteration')
    plt.ylabel(r'Error')
    # plt.legend(legend_plt, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid()
    plt.savefig(os.path.join(dir_path, 'epsilon_min_T_error.png'), dpi=500)


def create_sim_model(model_type, sim_param_dict, x_train, y_train, x_val, y_val, x_vec, y_vec, base_theta):
    if model_type == 'ECESA_GD':
        model = ECESAGDSimulator(sim_param_dict=sim_param_dict)
    elif model_type == 'ECESA_SGD':
        model = ECESASGDSimulator(sim_param_dict=sim_param_dict)
    elif model_type == 'ECESA_AGD':
        model = ECESAAGDSimulator(sim_param_dict=sim_param_dict)
    elif model_type == 'SGMA':
        model = GBMASimulator(sim_param_dict=sim_param_dict)
    elif model_type == 'GBMA':
        model = SGMASimulator(sim_param_dict=sim_param_dict)
    elif model_type == 'AGMA':
        model = AGMASimulator(sim_param_dict=sim_param_dict)
    elif model_type == 'FDM_GD':
        model = FDMGDSimulator(sim_param_dict=sim_param_dict)
    elif model_type == 'FDM_SGD':
        model = FDMSGDSimulator(sim_param_dict=sim_param_dict)
    elif model_type == 'FDM_AGD':
        model = FDMAGDSimulator(sim_param_dict=sim_param_dict)

    model.set_data(x_train, y_train, x_val, y_val, x_vec, y_vec)
    model.set_models()
    model.server_model.theta = np.copy(base_theta)
    model.server_model.theta_hist = np.copy(base_theta)
    return model


def run_process(model):
    model.run_full_simulation()


def simulate_noise(sigma_w, sigma_h, noise_size, worker_number, E_N):
    # making receiver noise
    w = [np.random.normal(0, sigma_w, noise_size) * np.sqrt(worker_number / E_N) for _ in range(worker_number)]
    # crating channel noise - federated effect:
    mu_h = 1 if sigma_h == 0 else np.sqrt(np.pi / (4 - np.pi)) * sigma_h
    if sigma_h == 0:
        h = np.ones(shape=(worker_number, 1))
    else:
        h = np.random.rayleigh(scale=np.sqrt(2 / np.pi) * mu_h, size=(worker_number, 1))
    # clip fading values to [0,1]
    h = np.clip(h, 0, 1)
    return h, w


def run_all_modes():
    process = []
    models = []
    threads = []
    with open("simulation_config.json", 'r') as f:
        sim_param_dict = json.load(f)

    if not sim_param_dict["data_path"]:
        if sim_param_dict["data_mode"]:
            path = os.path.join(os.getcwd(), 'data', r"YearPredictionMSD.txt")
        else:
            path = os.path.join(os.getcwd(), 'data', r"prices.csv")
    else:
        path = sim_param_dict["data_path"]

    if sim_param_dict["data_mode"]:
        x_train, y_train, x_val, y_val = load_data(data_path=path, data_ratio=sim_param_dict["data_ratio"])
    else:
        x_train, y_train, x_val, y_val = load_data2(data_path=path,
                                                    history_window_size=sim_param_dict["history_window_size"],
                                                    data_ratio=sim_param_dict["data_ratio"])

    x_vec, y_vec = data_split(x_train, y_train, sim_param_dict["worker_number"])
    base_theta = sim_param_dict["theta_start_amplitude"] * np.random.randn(x_train.shape[1], )
    if sim_param_dict["problem_type"] == "convex":
        N = 4
    else:
        N = 3
    for i in range(N):
        sim_param_dict["learning_rate_update_mode"] = i
        process.append(
            create_sim_model("SGMA", sim_param_dict, x_train, y_train, x_val, y_val, x_vec,
                             y_vec, base_theta))

    print('before thread loop')
    for i in range(len(process)):
        x = threading.Thread(target=run_process, args=(process[i],))
        threads.append(x)
        x.start()
    print('before thread join loop')

    for index, thread in enumerate(threads):
        thread.join()
    print('after thread join loop')

    for i in range(len(process)):
        # process[i].draw_results()
        process[i].server_model.default_dir = f"mode {i}"
        models.append(process[i].server_model)

    draw_all_results(models, sim_param_dict)
    print('done printing')


def run_multy_as_one():
    with open("simulation_config.json", 'r') as f:
        sim_param_dict = json.load(f)

    if not sim_param_dict["data_path"]:
        if sim_param_dict["data_mode"]:
            path = os.path.join(os.getcwd(), 'data', r"YearPredictionMSD.txt")
        else:
            path = os.path.join(os.getcwd(), 'data', r"prices.csv")
    else:
        path = sim_param_dict["data_path"]

    if sim_param_dict["data_mode"]:
        x_train, y_train, x_val, y_val = load_data(data_path=path, data_ratio=sim_param_dict["data_ratio"])
    else:
        x_train, y_train, x_val, y_val = load_data2(data_path=path,
                                                    history_window_size=sim_param_dict["history_window_size"],
                                                    data_ratio=sim_param_dict["data_ratio"])

    x_vec, y_vec = data_split(x_train, y_train, sim_param_dict["worker_number"])
    base_theta = sim_param_dict["theta_start_amplitude"] * np.random.randn(x_train.shape[1], )

    models = []
    for i in range(len(sim_param_dict['models_to_run'])):
        models.append(
            create_sim_model(sim_param_dict['models_to_run'][i], sim_param_dict, x_train, y_train, x_val, y_val, x_vec,
                             y_vec, base_theta))

    noise_size = models[0].server_model.grad.shape
    for time_index in tqdm(range(sim_param_dict['server_loop'])):
        channel_noise, receiver_noise = simulate_noise(sim_param_dict['sigma_w'], sim_param_dict['sigma_h'], noise_size,
                                                       sim_param_dict['worker_number'], sim_param_dict['E_N'])

        for model in models:
            model.one_iteration_running(time_index=time_index, channel_noise=channel_noise, receiver_noise=receiver_noise)

    for model in models:
        model.server_model.build_theta_hat()
        model.server_model.calculate_errors()

    models_server_list = []
    for model in models:
        models_server_list.append(model.server_model)
    draw_all_results(models_server_list, sim_param_dict)
    print('Simulation end')


def main():
    process = []
    models = []
    threads = []
    with open("simulation_config.json", 'r') as f:
        sim_param_dict = json.load(f)

    if not sim_param_dict["data_path"]:
        if sim_param_dict["data_mode"]:
            path = os.path.join(os.getcwd(), 'data', r"YearPredictionMSD.txt")
        else:
            path = os.path.join(os.getcwd(), 'data', r"prices.csv")
    else:
        path = sim_param_dict["data_path"]

    if sim_param_dict["data_mode"]:
        x_train, y_train, x_val, y_val = load_data(data_path=path, data_ratio=sim_param_dict["data_ratio"])
    else:
        x_train, y_train, x_val, y_val = load_data2(data_path=path,
                                                    history_window_size=sim_param_dict["history_window_size"],
                                                    data_ratio=sim_param_dict["data_ratio"])

    x_vec, y_vec = data_split(x_train, y_train, sim_param_dict["worker_number"])
    base_theta = sim_param_dict["theta_start_amplitude"] * np.random.randn(x_train.shape[1], )
    for i in range(len(sim_param_dict['models_to_run'])):
        process.append(
            create_sim_model(sim_param_dict['models_to_run'][i], sim_param_dict, x_train, y_train, x_val, y_val, x_vec,
                             y_vec, base_theta))

    print('before thread loop')
    for i in range(len(process)):
        x = threading.Thread(target=run_process, args=(process[i],))
        threads.append(x)
        x.start()
    print('before thread join loop')

    for index, thread in enumerate(threads):
        thread.join()
    print('after thread join loop')

    for i in range(len(process)):
        process[i].draw_results()
        models.append(process[i].server_model)

    draw_all_results(models, sim_param_dict)
    print('done printing')


if __name__ == '__main__':
    # main()
    run_multy_as_one()
    # run_all_modes()

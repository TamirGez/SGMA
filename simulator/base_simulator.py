from typing import List, Tuple, Optional
from models.base_model import BaseModel

import numpy as np
import time



class BaseSimulator:
    """
    A class that simulates a federated learning setup with a single server and multiple workers.
    """

    def __init__(self, sim_param_dict: dict) -> None:
        """
        Initializes the simulator with the given simulation parameters.

        Args:
            sim_param_dict (dict): A dictionary of simulation parameters.
        """
        self.sim_param_dict = sim_param_dict
        self.worker_number = sim_param_dict["worker_number"]
        self.server_loop = sim_param_dict["server_loop"]
        self.sigma_h = sim_param_dict["sigma_h"]
        self.sigma_w = sim_param_dict["sigma_w"]
        self.E_N = sim_param_dict["E_N"]
        self.eps = sim_param_dict["eps"]

        self.server_model = None
        self.workers: List[BaseModel] = []

    def set_data(self, x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray,
                 y_val: np.ndarray, x_vec: List[np.ndarray], y_vec: List[np.ndarray]) -> None:
        """
        Sets the data used for training and validation.

        Args:
            x_train (np.ndarray): The training input data.
            y_train (np.ndarray): The training output data.
            x_val (np.ndarray): The validation input data.
            y_val (np.ndarray): The validation output data.
            x_vec (List[np.ndarray]): A list of input data arrays for each worker.
            y_vec (List[np.ndarray]): A list of output data arrays for each worker.
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.x_vec = x_vec
        self.y_vec = y_vec

    def set_models(self) -> None:
        """
        Initializes the server and worker models.
        """
        mu_h = 1 if self.sigma_h == 0 else np.sqrt(np.pi / (4 - np.pi)) * self.sigma_h
        self.server_model = BaseModel(self.x_train, self.y_train, self.x_val, self.y_val, self.sim_param_dict)
        self.server_model.beta = 1 / (mu_h * self.server_model.L)

        for i in range(self.worker_number):
            new_worker = BaseModel(self.x_vec[i], self.y_vec[i], self.x_val, self.y_val, self.sim_param_dict)
            new_worker.beta = 1 / (mu_h * new_worker.L)
            self.workers.append(new_worker)

    def simulate_noise(self, receiver_noise_size: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates channel and receiver noise.

        Args:
            receiver_noise_size (int): The size of the receiver noise. Defaults to 1.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the channel noise and receiver noise arrays.
        """
        # making receiver noise
        if receiver_noise_size > 1:
            w = [
                np.random.normal(0, self.sigma_w, self.server_model.grad.shape) * np.sqrt(self.worker_number / self.E_N)
                for _ in range(receiver_noise_size)]
        else:
            w = np.random.normal(0, self.sigma_w, self.server_model.grad.shape) * np.sqrt(self.worker_number / self.E_N)
        # creating channel noise - federated effect:
        mu_h = 1 if self.sigma_h == 0 else np.sqrt(np.pi / (4 - np.pi)) * self.sigma_h
        if self.sigma_h == 0:
            h = np.ones(shape=(self.worker_number, 1))
        else:
            h = np.random.rayleigh(scale=np.sqrt(2 / np.pi) * mu_h, size=(self.worker_number, 1))
        h = np.clip(h, 0, 1)
        return h, w

    def one_iteration_running(self, time_index: int, channel_noise: Optional[np.ndarray] = None,
                              receiver_noise: Optional[np.ndarray] = None) -> None:
        """
        Runs a single iteration of the federated learning algorithm.

        Args:
            time_index (int): The current time index.
            channel_noise (Optional[np.ndarray]): The channel noise. Defaults to None.
            receiver_noise (Optional[np.ndarray]): The receiver noise. Defaults to None.
        """
        self.server_model.grad[:] = 0
        # building channel noise
        if channel_noise is None:
            channel_noise, receiver_noise = self.simulate_noise()
        else:
            receiver_noise = receiver_noise[0] if isinstance(receiver_noise, list) else receiver_noise
        for i in range(self.worker_number):
            # updating for each worker the theta and gamma as in the server
            self.workers[i].theta_hist = np.copy(self.workers[i].theta)
            self.workers[i].theta = np.copy(self.server_model.theta)
            self.workers[i].learning_rate_pre = np.copy(self.workers[i].learning_rate)
            self.workers[i].learning_rate = np.copy(self.server_model.learning_rate)
            # local iteration step
            self.workers[i].calculate_z()
            self.workers[i].calculate_grad()
            # federated effect here
            self.server_model.grad += channel_noise[i] * np.copy(self.workers[i].grad)

        # final server gradient update
        self.server_model.grad = self.server_model.grad + receiver_noise
        self.server_model.grad = self.server_model.grad / self.worker_number
        self.server_model.grad[np.abs(self.server_model.grad) < self.eps] = 0
        self.server_model.update_theta_process(time_index=time_index)

    def run_full_simulation(self) -> None:
        """
        Runs the full federated learning simulation.
        """
        t0 = time.time()
        for time_index in range(self.server_loop):
            self.one_iteration_running(time_index=time_index)
        tn = time.time()
        print('model ' + self.server_model.default_dir + ' as finished, train time=' + str(tn - t0))
        self.server_model.build_theta_hat()
        self.server_model.calculate_errors()

    def draw_results(self) -> None:
        """
        Plots and saves the results of the simulation.
        """
        self.server_model.plot_results(start_skip=self.sim_param_dict["start_skip"],
                                       save_results=self.sim_param_dict["save_results"],
                                       save_theta_history=self.sim_param_dict["save_theta_history"],
                                       save_dir=self.sim_param_dict["save_dir"])

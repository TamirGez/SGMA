from simulator.base_simulator import BaseSimulator
from models.models import GradientDescentModel, SubGradientDescentModel, AcceleratedGradientDescentModel
import numpy as np


class FDMGDSimulator(BaseSimulator):
    def __init__(self, sim_param_dict):
        super().__init__(sim_param_dict)

    def set_models(self):
        mu_h = 1 if self.sigma_h == 0 else np.sqrt(np.pi / (4 - np.pi)) * self.sigma_h
        self.server_model = GradientDescentModel(self.x_train, self.y_train, self.x_val, self.y_val, self.sim_param_dict)
        self.server_model.beta = 1 / (mu_h * self.server_model.L)
        self.server_model.default_dir = 'FDM_GD'

        for i in range(self.worker_number):
            new_worker = GradientDescentModel(self.x_vec[i], self.y_vec[i], self.x_val, self.y_val, self.sim_param_dict)
            new_worker.beta = 1 / (mu_h * new_worker.L)
            self.workers.append(new_worker)

    def one_iteration_running(self, time_index, channel_noise=None, receiver_noise=None):
        self.server_model.grad[:] = 0
        # building channel noise
        if channel_noise is None:
            channel_noise, receiver_noise = self.simulate_noise(self.worker_number)

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
            self.server_model.grad += channel_noise[i] * np.copy(self.workers[i].grad) + receiver_noise[i]

        # final server gradient update
        self.server_model.grad = self.server_model.grad / self.worker_number
        self.server_model.grad[np.abs(self.server_model.grad) < self.eps] = 0
        self.server_model.update_theta_process(time_index=time_index)


class FDMSGDSimulator(BaseSimulator):
    def __init__(self, sim_param_dict):
        super().__init__(sim_param_dict)

    def set_models(self):
        mu_h = 1 if self.sigma_h == 0 else np.sqrt(np.pi / (4 - np.pi)) * self.sigma_h
        self.server_model = SubGradientDescentModel(self.x_train, self.y_train, self.x_val, self.y_val, self.sim_param_dict)
        self.server_model.beta = 1 / (mu_h * self.server_model.L)
        self.server_model.default_dir = 'FDM_SGD'

        for i in range(self.worker_number):
            new_worker = SubGradientDescentModel(self.x_vec[i], self.y_vec[i], self.x_val, self.y_val, self.sim_param_dict)
            new_worker.beta = 1 / (mu_h * new_worker.L)
            self.workers.append(new_worker)

    def one_iteration_running(self, time_index, channel_noise=None, receiver_noise=None):
        self.server_model.grad[:] = 0
        # building channel noise
        if channel_noise is None:
            channel_noise, receiver_noise = self.simulate_noise(self.worker_number)

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
            self.server_model.grad += channel_noise[i] * np.copy(self.workers[i].grad) + receiver_noise[i]

        # final server gradient update
        self.server_model.grad = self.server_model.grad / self.worker_number
        self.server_model.grad[np.abs(self.server_model.grad) < self.eps] = 0
        self.server_model.update_theta_process(time_index=time_index)


class FDMAGDSimulator(BaseSimulator):
    def __init__(self, sim_param_dict):
        super().__init__(sim_param_dict)

    def set_models(self):
        mu_h = 1 if self.sigma_h == 0 else np.sqrt(np.pi / (4 - np.pi)) * self.sigma_h
        self.server_model = AcceleratedGradientDescentModel(self.x_train, self.y_train, self.x_val, self.y_val,
                                                         self.sim_param_dict)
        self.server_model.beta = 1 / (mu_h * self.server_model.L)
        self.server_model.default_dir = 'FDM_AGD'
        for i in range(self.worker_number):
            new_worker = AcceleratedGradientDescentModel(self.x_vec[i], self.y_vec[i], self.x_val, self.y_val,
                                                      self.sim_param_dict)
            new_worker.beta = 1 / (mu_h * new_worker.L)
            self.workers.append(new_worker)

    def one_iteration_running(self, time_index, channel_noise=None, receiver_noise=None):
        self.server_model.grad[:] = 0
        # building channel noise
        if channel_noise is None:
            channel_noise, receiver_noise = self.simulate_noise(self.worker_number)

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
            self.server_model.grad += channel_noise[i] * np.copy(self.workers[i].grad) + receiver_noise[i]

        # final server gradient update
        self.server_model.grad = self.server_model.grad / self.worker_number
        self.server_model.grad[np.abs(self.server_model.grad) < self.eps] = 0
        self.server_model.update_theta_process(time_index=time_index)

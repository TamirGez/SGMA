from simulator.base_simulator import BaseSimulator
from models.models import GradientDescentModel, SubGradientDescentModel, AcceleratedGradientDescentModel
import numpy as np


class GBMASimulator(BaseSimulator):
    def __init__(self, sim_param_dict):
        super().__init__(sim_param_dict)

    def set_models(self):
        mu_h = 1 if self.sigma_h == 0 else np.sqrt(np.pi / (4 - np.pi)) * self.sigma_h
        self.server_model = GradientDescentModel(self.x_train, self.y_train, self.x_val, self.y_val,
                                                 self.sim_param_dict)
        self.server_model.beta = 1 / (mu_h * self.server_model.L)

        for i in range(self.worker_number):
            new_worker = GradientDescentModel(self.x_vec[i], self.y_vec[i], self.x_val, self.y_val, self.sim_param_dict)
            new_worker.beta = 1 / (mu_h * new_worker.L)
            self.workers.append(new_worker)


class SGMASimulator(BaseSimulator):
    def __init__(self, sim_param_dict):
        super().__init__(sim_param_dict)

    def set_models(self):
        mu_h = 1 if self.sigma_h == 0 else np.sqrt(np.pi / (4 - np.pi)) * self.sigma_h
        self.server_model = SubGradientDescentModel(self.x_train, self.y_train, self.x_val, self.y_val,
                                                    self.sim_param_dict)
        self.server_model.beta = 1 / (mu_h * self.server_model.L)

        for i in range(self.worker_number):
            new_worker = SubGradientDescentModel(self.x_vec[i], self.y_vec[i], self.x_val, self.y_val,
                                                 self.sim_param_dict)
            new_worker.beta = 1 / (mu_h * new_worker.L)
            self.workers.append(new_worker)


class AGMASimulator(BaseSimulator):
    """
    A subclass of BaseSimulator that implements the AGMA algorithm for federated learning.
    """

    def __init__(self, sim_param_dict):
        super().__init__(sim_param_dict)

    def set_models(self):
        mu_h = 1 if self.sigma_h == 0 else np.sqrt(np.pi / (4 - np.pi)) * self.sigma_h
        self.server_model = AcceleratedGradientDescentModel(self.x_train, self.y_train, self.x_val, self.y_val,
                                                            self.sim_param_dict)
        self.server_model.beta = 1 / (mu_h * self.server_model.L)

        for i in range(self.worker_number):
            new_worker = AcceleratedGradientDescentModel(self.x_vec[i], self.y_vec[i], self.x_val, self.y_val,
                                                         self.sim_param_dict)
            new_worker.beta = 1 / (mu_h * new_worker.L)
            self.workers.append(new_worker)

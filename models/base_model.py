import json
import os
import shutil
import numpy as np
import pylab as plt


class BaseModel:
    """Base model for gradient descent model."""

    def __init__(self, x_train, y_train, x_val, y_val, sim_params):
        """
        Initializes BaseModel.

        Args:
            x_train (numpy.ndarray): Training data features.
            y_train (numpy.ndarray): Training data labels.
            x_val (numpy.ndarray): Validation data features.
            y_val (numpy.ndarray): Validation data labels.
            sim_params (dict): Dictionary containing simulation parameters.
        """
        # Set input and running params.
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        self.sim_params = sim_params
        self.learning_rate = sim_params["learning_rate"]
        self.elastic_lasso = sim_params["elastic_lasso"]
        self.elastic_bright = sim_params["elastic_bright"]
        self.problem_type = sim_params["problem_type"]
        self.learning_rate_update_mode = sim_params["learning_rate_update_mode"]
        self.eps = sim_params["eps"]
        self.theta_start_amplitude = sim_params["theta_start_amplitude"]
        self.max_epochs = sim_params["max_epochs"]
        self.huber_mu = sim_params["huber_mu"]
        self.with_update_eta = False

        # Initialize run params.
        self.default_dir = 'base'
        self.mu = self.elastic_bright / 2
        self.theta = self.theta_start_amplitude * np.random.randn(x_train.shape[1], )
        self.grad = np.zeros(x_train.shape[1], )
        self.L = self._calculate_L(x_train, self.mu)
        self.z = np.copy(self.theta)
        self.learning_rate_pre = np.copy(self.learning_rate)
        self.theta_hist = np.copy(self.theta)
        self.beta = 1 / self.L
        self.eta = 0
        self.learning_rate0 = self.learning_rate

        self.f_vals = []
        self._calculate_f_vals()

        self.theta_history = []
        self.grad_history = []
        self.theta_hat_with_history = []
        self.X_hat_top = [0]
        self.X_hat_bottom = [0]
        self.error_rate_mse = []
        self.error_rate_mse_train = []
        self.error_rate_elastic_mse = []
        self.error_rate_elastic_mse_train = []
        self.epsilon_error_theta_hat = []
        self.epsilon_error_min_T = []

    @staticmethod
    def _calculate_L(x_train, mu):
        """
        Calculates L for gradient descent.

        Args:
            x_train (numpy.ndarray): Training data features.
            mu (float): Value for mu.

        Returns:
            float: Value of L.
        """
        lin_reg = 0
        for i in range(x_train.shape[0]):
            lin_reg += np.dot(x_train[i].reshape((-1, 1)), x_train[i].reshape(1, -1))
        lin_reg = lin_reg / x_train.shape[0]
        lin_reg += mu * np.eye(lin_reg.shape[0])
        return np.linalg.norm(lin_reg, 2)

    def _update_eta(self):
        """
        Updates eta value for gradient descent.
        """
        alpha_k = self.learning_rate_pre
        alpha_k1 = self.learning_rate
        self.eta = alpha_k * (1 - alpha_k) / (alpha_k1 + alpha_k ** 2)
        self.learning_rate_pre = np.copy(self.learning_rate)

    def calculate_z(self):
        """
        calculate z for learning process, this used for acceleration model.
        if we are not on acceleration model, then we will set eta to 1 with igrnore acceleration.
        """
        if self.with_update_eta:
            self._update_eta()
        self.z = self.theta - self.eta * (self.theta - self.theta_hist)
        self.z[np.abs(self.z) < self.eps] = 0

    def update_theta(self):
        """
        saving current weights of the model, and doing learning rate step.
        """
        self.theta_hist = np.copy(self.theta)
        self.theta = self.z - self.learning_rate * self.grad
        self.theta[np.abs(self.theta) < self.eps] = 0
        self.calculate_z()

    def huber_function(self, param):
        """
        This is an implementation of the Hoover function.
        """
        huber_f = (param ** 2) / (2 * self.huber_mu)
        idx = np.abs(param) > self.huber_mu
        huber_f[idx] = np.abs(param[idx]) - self.huber_mu / 2
        return np.sum(huber_f)

    def huber_function_divergent(self, param):
        """
        This is an implementation of the Hoover function divergent
        """
        dfdx = np.sign(param)
        idx = np.abs(param) < self.huber_mu
        dfdx[idx] = param[idx] / self.huber_mu
        return dfdx

    def _calculate_f_vals(self):
        """
        Calculates the objective function for each iteration of gradient descent.
        """
        f1 = (1 / (2 * self.x_train.shape[0])) * (np.linalg.norm(self.x_train @ self.theta - self.y_train, 2) ** 2)
        f2 = self.elastic_lasso * self.huber_function(self.theta)
        f3 = (self.elastic_bright / 2) * (np.linalg.norm(self.theta, 2) ** 2)
        current_f_vals = f1 + f2 + f3
        self.f_vals.append(current_f_vals)

    def calculate_grad(self):
        grad1 = (1 / self.x_train.shape[0]) * (self.x_train @ self.z - self.y_train) @ self.x_train
        grad2 = self.elastic_lasso * self.huber_function_divergent(self.z)
        grad3 = self.elastic_bright * self.z
        self.grad = grad1 + grad2 + grad3
        self.grad[np.abs(self.grad) < self.eps] = 0

    def update_learning_rate(self, idx):
        if self.problem_type == 'convex':
            if self.learning_rate_update_mode == 0:
                pass
            elif self.learning_rate_update_mode == 1:
                self.learning_rate = self.learning_rate0 / np.linalg.norm(self.grad, 2)
            elif self.learning_rate_update_mode == 2:
                self.learning_rate = self.learning_rate0 / np.sqrt(idx + 1)
            elif self.learning_rate_update_mode == 3:
                self.learning_rate = self.learning_rate0 / (idx + 1)
        elif self.problem_type == 'strongly_convex':
            if self.learning_rate_update_mode == 0:
                pass
            elif self.learning_rate_update_mode == 1:
                self.learning_rate = 1 / (self.mu * (idx + 1))
            elif self.learning_rate_update_mode == 2:
                self.learning_rate = 2 / (self.mu * (idx + 2))

    def update_theta_process(self, time_index):
        self.update_theta()
        self._calculate_f_vals()
        self.theta_history.append(np.copy(self.theta))
        self.grad_history.append(np.copy(self.grad))
        self.X_hat_top.append(self.X_hat_top[-1] + np.copy(self.theta) * self.learning_rate)
        self.X_hat_bottom.append(self.X_hat_bottom[-1] + self.learning_rate)
        self.update_learning_rate(time_index)

    def build_theta_hat(self):
        self.theta_hat_with_history = [self.X_hat_top[i] / self.X_hat_bottom[i] for i in range(1, len(self.X_hat_top))]

    def find_F_min_by_index(self, idx):
        return np.min(self.f_vals[:(idx + 1)])

    def calc_F_for_theta_hat(self, theta_hat):
        F1 = (1 / (2 * self.x_val.shape[0])) * (np.linalg.norm(self.x_val @ theta_hat - self.y_val, 2) ** 2)
        F2 = self.elastic_lasso * self.huber_function(theta_hat)
        F3 = (self.elastic_bright / 2) * (np.linalg.norm(theta_hat, 2) ** 2)
        return F1 + F2 + F3

    def calculate_mse(self, theta, xdata, ydata):
        return (1 / (2 * xdata.shape[0])) * (np.linalg.norm(xdata @ theta - ydata, 2) ** 2)

    def calculate_elastic_mse(self, theta, xdata, ydata):
        mse = self.calculate_mse(theta, xdata, ydata)
        lasso = self.elastic_lasso * np.linalg.norm(theta, 1)
        bright = ((self.elastic_bright / 2) * (np.linalg.norm(theta, 2) ** 2))
        elastic_mse = mse + lasso + bright
        return elastic_mse

    def calculate_errors(self):
        self.error_rate_elastic_mse = np.array(
            [self.calculate_elastic_mse(self.theta_history[i], self.x_val, self.y_val) for i in
             range(len(self.theta_history))])
        self.error_rate_elastic_mse_train = np.array(
            [self.calculate_elastic_mse(self.theta_history[i], self.x_train, self.y_train) for i in
             range(len(self.theta_history))])
        self.error_rate_mse = np.array(
            [self.calculate_mse(self.theta_history[i], self.x_val, self.y_val) for i in range(len(self.theta_history))])
        self.error_rate_mse_train = np.array(
            [self.calculate_mse(self.theta_history[i], self.x_train, self.y_train) for i in
             range(len(self.theta_history))])
        self.epsilon_error_theta_hat = np.array(
            [np.abs(self.calc_F_for_theta_hat(self.theta_hat_with_history[i])) for i in
             range(len(self.theta_hat_with_history))])
        self.epsilon_error_min_T = np.array([np.abs(self.find_F_min_by_index(i)) for i in range(1, len(self.f_vals))])

    def plot_results(self, start_skip=0, save_results=False, save_theta_history=False, save_dir=None):
        if save_results:
            if not save_dir:
                dir_path = os.path.join(os.getcwd(), self.default_dir)
            else:
                dir_path = os.path.join(os.getcwd(), save_dir)
            if os.path.isdir(dir_path):
                shutil.rmtree(dir_path)
            os.mkdir(dir_path)
            with open(os.path.join(dir_path, 'sim_info.json'), 'w') as f:
                json.dump(self.sim_param_dict, f)

            if save_theta_history:
                np.save(os.path.join(dir_path, "model_theta_history.npy"), self.theta_history)

        plt.figure(1)
        plt.clf()
        plt.semilogy(np.arange(start_skip, self.epsilon_error_theta_hat.shape[0]),
                     self.epsilon_error_theta_hat[start_skip:])
        plt.title('error as theta_hat')
        plt.xlabel('iteration [k]')
        # plt.ylabel(r'$\mathbb{E}' r'[F(\boldsymbol{\hat{\theta}_{T}})-F(\boldsymbol{\theta^{\ast})}]$')
        if save_results:
            plt.savefig(os.path.join(dir_path, 'Theta_hat_error.png'))
        # plt.show()

        plt.figure(2)
        plt.clf()
        plt.semilogy(np.arange(start_skip, self.error_rate_mse.shape[0]), self.error_rate_mse[start_skip:])
        plt.title('error as MSE')
        if save_results:
            plt.savefig(os.path.join(dir_path, 'MSE_error.png'))
        # plt.show()

        plt.figure(3)
        plt.clf()
        plt.semilogy(np.arange(start_skip, self.error_rate_elastic_mse.shape[0]),
                     self.error_rate_elastic_mse[start_skip:])
        plt.title('error as elastic MSE')
        if save_results:
            plt.savefig(os.path.join(dir_path, 'elastic_MSE_error.png'))
        # plt.show()

        plt.figure(4)
        plt.clf()
        plt.semilogy(np.arange(start_skip, self.epsilon_error_min_T.shape[0]), self.epsilon_error_min_T[start_skip:])
        plt.title('epsilon min T error')
        if save_results:
            plt.savefig(os.path.join(dir_path, 'epsilon_min_T_error.png'))

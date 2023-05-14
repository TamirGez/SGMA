from models.base_model import BaseModel
import numpy as np


class GradientDescentModel(BaseModel):
    def __init__(self, x_train, y_train, x_val, y_val, sim_param_dict):
        super().__init__(x_train, y_train, x_val, y_val, sim_param_dict)
        self.default_dir = "GBMA"


class SubGradientDescentModel(BaseModel):
    def __init__(self, x_train, y_train, x_val, y_val, sim_param_dict):
        super().__init__(x_train, y_train, x_val, y_val, sim_param_dict)
        self.default_dir = "SGMA"
        self.possible_values_list = np.linspace(-1, 1, 101).tolist()

    def huber_function(self, param):
        return np.linalg.norm(param, 1)

    def huber_function_divergent(self, param):
        dfdx = np.sign(param)
        return dfdx

    def calculate_grad(self):
        grad1 = (1 / self.x_train.shape[0]) * (self.x_train @ self.z - self.y_train) @ self.x_train
        grad2 = self.huber_function_divergent(self.z)
        grad3 = self.elastic_bright * self.z
        if np.any(grad2 == 0):
            zero_indices = np.where(grad2 == 0)[0]
            best_score = float('inf')
            dfdx_copy = grad2.copy()
            best_dfdx = dfdx_copy.copy()
            # Iterate over all combinations of possible values for each zero index
            for combination in np.ndindex(*([len(self.possible_values_list)] * len(zero_indices))):
                # Modify the array copy with the current combination of values
                for index, value in zip(zero_indices, combination):
                    dfdx_copy[index] = self.possible_values_list[value]
                # Calculate the score for the modified array
                temp_theta = self.z - (self.elastic_lasso * dfdx_copy + grad1 + grad3) * self.learning_rate
                score = self.calculate_elastic_mse(temp_theta, self.x_val, self.y_val)
                # Update the best score and corresponding array if the current score is better
                if score < best_score:
                    best_score = score
                    best_dfdx = dfdx_copy.copy()
                # Restore the original array copy
                dfdx_copy = grad2.copy()
            grad2 = self.elastic_lasso * best_dfdx

        self.grad = grad1 + grad2 + grad3
        self.grad[np.abs(self.grad) < self.eps] = 0


class AcceleratedGradientDescentModel(BaseModel):
    """
    A subclass of BaseModel that implements the accelerated gradient descent algorithm.
    """

    def __init__(self, x_train, y_train, x_val, y_val, sim_param_dict):
        super().__init__(x_train, y_train, x_val, y_val, sim_param_dict)
        self.default_dir = "AGMA"
        self.with_update_eta = True

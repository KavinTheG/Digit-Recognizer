import numpy as np

class LossCategoricalEntropy:

    def __init__(self, y_hat, y_target):
        self.y_target = y_target
        self.y_hat = y_hat

    def calculate_loss(self):
        
        log_values = np.log(self.y_hat)

        #loss_i = np.dot(log_values, self.y_target.T)

        #self.result = -np.sum(loss_i)

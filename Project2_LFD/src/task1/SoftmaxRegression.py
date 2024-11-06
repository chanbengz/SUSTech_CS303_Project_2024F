import numpy as np
from tqdm import tqdm

class SoftmaxRegression:
    def __init__(
        self, 
        num_classes, 
        learning_rate=0.01, 
        num_iterations=100, 
        random_seed=None,
        optimizer='gd', 
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.random_seed = random_seed
        self.optimizer = optimizer.lower()
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weights = None
        self.m = None
        self.v = None
        self.t = 0  

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_bias = np.hstack((np.ones((X_train.shape[0], 1)), X_train))
        np.random.seed(self.random_seed)
        self.weights = np.random.randn(X_train_bias.shape[1], self.num_classes)

        if self.optimizer == 'adam':
            self.m = np.zeros_like(self.weights)
            self.v = np.zeros_like(self.weights)
            self.t = 0 

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for iteration in tqdm(range(1, self.num_iterations + 1)):
            logits = np.dot(X_train_bias, self.weights)
            exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # 稳定性处理
            softmax_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            
            loss = -np.mean(y_train * np.log(softmax_probs + 1e-15)) 
            
            # 计算梯度
            gradient = np.dot(X_train_bias.T, (softmax_probs - y_train)) / X_train_bias.shape[0]

            if self.optimizer == 'adam':
                self.t += 1
                self.m = self.beta1 * self.m + (1 - self.beta1) * gradient
                self.v = self.beta2 * self.v + (1 - self.beta2) * (gradient ** 2)
                m_hat = self.m / (1 - self.beta1 ** self.t)
                v_hat = self.v / (1 - self.beta2 ** self.t)
                self.weights -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
            else:
                self.weights -= self.learning_rate * gradient

            train_pred = np.argmax(softmax_probs, axis=1)
            train_accuracy = np.mean(train_pred == np.argmax(y_train, axis=1))
            
            train_losses.append(loss)
            train_accuracies.append(train_accuracy)

            if X_val is not None and y_val is not None:
                X_val_bias = np.hstack((np.ones((X_val.shape[0], 1)), X_val))
                logits_val = np.dot(X_val_bias, self.weights)
                exp_logits_val = np.exp(logits_val - np.max(logits_val, axis=1, keepdims=True))
                softmax_probs_val = exp_logits_val / np.sum(exp_logits_val, axis=1, keepdims=True)
                val_loss = -np.mean(y_val * np.log(softmax_probs_val + 1e-15))
                val_pred = np.argmax(softmax_probs_val, axis=1)
                val_accuracy = np.mean(val_pred == np.argmax(y_val, axis=1))
                
                val_losses.append(val_loss)
                val_accuracies.append(val_accuracy)

        return train_losses, val_losses, train_accuracies, val_accuracies

    def predict(self, X):
        """
        使用训练好的模型进行预测。

        参数:
        - X: 预测所需的特征数据。

        返回:
        - predicted_class: 预测的类别标签。
        """
        X_bias = np.hstack((np.ones((X.shape[0], 1)), X))
        logits = np.dot(X_bias, self.weights)
        predicted_class = np.argmax(logits, axis=1)
        return predicted_class

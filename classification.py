# Copyright 2025 Divya Prakash Singh
# Licensed under the Apache License, Version 2.0
# See the LICENSE file for more details.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
import warnings

warnings.filterwarnings('ignore', category=RuntimeWarning)
np.set_printoptions(suppress=True, precision=6, threshold=np.inf)

class model:
    def __init__(self, featureSize, epoch, learningRate_weight, learningRate_bias, batchSize,
                 positive_strata_size, negative_strata_size, 
                 momentum_coefficient, regularization_coff, decision_threshold, yeta_elastic_net, optimizer_class):
        self.featureSize = featureSize
        self.epoch = epoch

        #Weight initialization and feature split into weights and bias.
        lower, upper = -(1.0 / np.sqrt(featureSize + 1)), (1.0 / np.sqrt(featureSize + 1))
        random_features = np.random.rand(featureSize + 1)
        self.features_combined = lower + random_features * (upper - lower)
        self.weights = self.features_combined[:-1]
        self.bias = self.features_combined[-1]
        #Initialization section completed

        self.learningRate_weight = learningRate_weight
        self.learningRate_bias = learningRate_bias
        self.batchSize = batchSize
        self.Positive_strata_size = positive_strata_size
        self.Negative_strata_size = negative_strata_size
        self.loss_history = []
        self.momentum_coff = momentum_coefficient
        self.velocity_weight = 0
        self.velocity_bias = 0
        self.regularization_coff = regularization_coff
        self.decision_threshold = decision_threshold
        self.yeta_elastic_net = yeta_elastic_net

        self.optimizer_class = optimizer_class
        self.optimizer_class.attach_model(self)

    #Regulariation section including the Ridge, Lasso and elastic net regularization.
    def elastic_net_regularization(self):
        #Ridge regularization
        squaredWeight = np.mean(self.weights ** 2)
        ridge_penalty = self.regularization_coff * squaredWeight

        #Lasso regularization
        absWeight = np.mean(self.weights)
        lasso_penalty = self.regularization_coff * absWeight

        #yeta is the hyperparameter for the elastic net regularization. On 0 it will be Lasso and on 1 it will be Ridge.
        penalty = (self.yeta_elastic_net * ridge_penalty) + ((1-self.yeta_elastic_net) * lasso_penalty)
        return penalty
    #Regularization section ends here.


    def linear_combination(self, feature):
        #print(f"Feature: {feature}\n")
        #print(f"Weights: {self.weights}\n")
        feature = np.array(feature)
        linear_combo = np.dot(feature, self.weights) + self.bias
        return linear_combo
    
    def sigmoidFunction(self, linear_combo):
        sigmoid = 1 / (1 + np.exp(-linear_combo))
        predictionValue = np.where(sigmoid >= self.decision_threshold, 1, 0)
        predictionValue = np.clip(sigmoid, 1e-10, 1 - 1e-10)
        return predictionValue
    
    def error(self, predictedValue, actualValue):
        predictedValue = np.clip(predictedValue, 1e-10, 1 - 1e-10)
        error = -np.mean((actualValue * np.log(predictedValue)) + ((1 - actualValue) * np.log(1 - predictedValue)))
        penalized_error = error + self.elastic_net_regularization()
        return penalized_error

    def training(self, training_dataset):
        valuePrediction = []
        num_samples = len(training_dataset)
        POSITIVE_strata = training_dataset[training_dataset[:, -1] == 1]
        NEGATIVE_strata = training_dataset[training_dataset[:, -1] == 0]

        for epoch in range(self.epoch):
            #Stratified sampling approach for balanced training on imbalanced dataset
            POSITIVE_batch = POSITIVE_strata[np.random.choice(POSITIVE_strata.shape[0], self.Positive_strata_size, replace=False)]
            NEGATIVE_batch = NEGATIVE_strata[np.random.choice(NEGATIVE_strata.shape[0], self.Negative_strata_size, replace=False)]
            training_data_combined = np.vstack((POSITIVE_batch, NEGATIVE_batch))

            trainingData = training_data_combined[:, 0:-1]
            trainingData_actualResult = training_data_combined[:, -1]

            linearCombinationValue = self.linear_combination(trainingData)
            valuePrediction = self.sigmoidFunction(linearCombinationValue)
            errorValue = self.error(valuePrediction, trainingData_actualResult)
            self.optimizer_class.gradientDescent(trainingData_actualResult, valuePrediction, trainingData)

            #Velocity section (DISABLED)
            '''velocityWeight = np.float32((self.momentum_coff * self.velocity_weight) + (self.learningRate_weight * delta_weight))
            velocityBias = np.float32((self.momentum_coff * self.velocity_bias) + (self.learningRate_bias * delta_bias))
        
            #Updating weights and bias value according to the gradient descent
            self.weights -= velocityWeight
            self.bias -= velocityBias'''
            self.loss_history.append(errorValue)

        print("Training completed\n")
        print(f"Cross-entropy loss Error on last Epoch: {errorValue}\n")
        print(f"Final weigths: {self.weights}\n")
        print(f"Final bias: {self.bias}\n")

    def evaluation(self, datasetFeature, actualResult):
        linearCombinationValue = self.linear_combination(datasetFeature)
        valuePrediction = self.sigmoidFunction(linearCombinationValue)
        check = pd.DataFrame({"Predicted": valuePrediction, "Actual": actualResult})
        check.to_csv("Predicted.csv", index=False)
        valuePrediction = np.where(valuePrediction >= self.decision_threshold, 1, 0)

        errorValue = self.error(valuePrediction, actualResult)
        target_mean = np.mean(actualResult)
        LL_null = self.error(target_mean, actualResult)
        actualResult = np.array(actualResult)

        accuracy = metrics.accuracy_score(actualResult, valuePrediction)
        precision = metrics.precision_score(actualResult, valuePrediction)
        recall = metrics.recall_score(actualResult, valuePrediction)
        f1_score = metrics.f1_score(actualResult, valuePrediction)
        confusionMatrix = metrics.confusion_matrix(actualResult, valuePrediction)

        return accuracy, precision, recall, f1_score, confusionMatrix, errorValue, LL_null

    def plot_loss(self, lr):
        plt.plot(self.loss_history)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        #fileName = "loss-over-epoch_polt-LR" + str(lr)
        #plt.savefig(fileName)
        plt.show()

    def retrive_loss(self):
        return self.loss_history

    def __del__(self):
        print("Deleting Main model Object.")

class Optimizer():
    def __init__(self, feature_Size, optimizer_using, RMS_beta2, adam_beta1, adam_beta2):

        self.RMS_beta2 = RMS_beta2
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.v_weighted_norm = np.zeros(feature_Size + 1)
        self.moment_vector = np.zeros(feature_Size + 1)
        self.optimizer_using = optimizer_using

        #All the decay parameters for the optimizer.
        self.m_hat = 0
        self.v_hat = 0
        self.rate_decay = 0
        #Decay section ends here.

        self.epsilon = 1e-8
        self.gradient_weights = 0
        self.gradient_bias = 0
        self.gradient_combined = 0
        self.model = None

    def attach_model(self, model_instance):
        self.model = model_instance

    def gradientDescent(self, actualValue, predictedValue, trainingData):
        dataSize = trainingData.shape[1]
        actualTrueLoss = predictedValue - actualValue

        self.gradient_weights = (np.dot(trainingData.T, actualTrueLoss) / dataSize) + ((self.model.regularization_coff * self.model.weights) / dataSize)
        self.gradient_bias = np.mean(actualTrueLoss)
        self.gradient_combined = np.concatenate((self.gradient_weights, np.array([self.gradient_bias])))
        if self.optimizer_using == 'ADAM':
            self.adam()
        elif self.optimizer_using == 'RMSprop':
            self.RMS_prop()
        return
        #return gradient_weights, gradient_bias
    
    def RMS_prop(self):
        self.v_weighted_norm = (self.RMS_beta2 * self.v_weighted_norm) + ((1 - self.RMS_beta2) * np.square(self.gradient_combined))
        self.model.features_combined -= (self.model.learningRate_weight / (np.sqrt(self.v_weighted_norm)) + self.epsilon) * self.gradient_combined
        self.model.weights = self.model.features_combined[:-1]
        self.model.bias = self.model.features_combined[-1] 
        return
    
    def decay(self):
        beta1_decay = self.adam_beta1 ** self.model.epoch
        beta2_decay = self.adam_beta2 ** self.model.epoch
        self.m_hat = self.moment_vector / (1 - beta1_decay)
        self.v_hat = self.v_weighted_norm / (1 - beta2_decay)
        self.rate_decay = self.model.learningRate_weight * ((np.sqrt(1 - beta2_decay)) / 1 - beta1_decay)
        return
    
    def adam(self):
        self.moment_vector = (self.adam_beta1 * self.moment_vector) + ((1 - self.adam_beta1) * self.gradient_combined)
        self.v_weighted_norm = (self.adam_beta2 * self.v_weighted_norm) + ((1 - self.adam_beta2) * np.square(self.gradient_combined))
        self.decay()
        self.model.features_combined -= self.rate_decay * (self.m_hat / (np.sqrt(self.v_hat) + self.epsilon))
        self.model.weights = self.model.features_combined[:-1]
        self.model.bias = self.model.features_combined[-1]
        return
   
    def __del__(self):
        print("Deleting Optimizer Object.")

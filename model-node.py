# Copyright 2025 Divya Prakash Singh
# Licensed under the Apache License, Version 2.0
# See the LICENSE file for more details.

import sys
sys.path.append(r"C:\AI-ML\projects\util")

import model_report_generator
import pandas as pd
import numpy as np
import classification
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from scipy.stats import chi2

filePath = "dataset/breast_cancer_PREPARED.csv"
dataset = pd.read_csv(filePath, encoding='UTF-8')

datasetSize = dataset.shape[0]
featureSize = dataset.shape[1] - 1

dataset = dataset.sample(frac=1).reset_index(drop=True)

trainingSize = int(0.8 * datasetSize)
testingSize = datasetSize - trainingSize

trainingData = dataset.iloc[:trainingSize, 0:-1]
trainingData_actualResult = dataset.iloc[:trainingSize, -1]
trainingData_actualResult = trainingData_actualResult.values.reshape(-1, 1)

evaulationData = dataset.iloc[trainingSize:, 0:-1]
evaluationData_actualResult = dataset.iloc[trainingSize:, -1]

scaler = StandardScaler()
trainingData = scaler.fit_transform(trainingData)
evaulationData = scaler.transform(evaulationData)
trainingData = np.hstack((trainingData, trainingData_actualResult))

#HyperParameters section -----Consists of all the hyperparameter values for the model control.
model_name = "Stroke Risk Prediction Model"
decision_threshold = 0.5
epoch = 5000
learningRate_weight = 0.01
learningRate_bias = 0.1
batchSize = 'NA'
POSITIVE_strata_size = 50
NEGATIVE_strata_size = 50
momentum_coff = 0.99
regularization_coff = 0.2
yeta_elastic_net = 1                                            #0 it will be Lasso and on 1 it will be Ridge.
RMS_beta1 = 0.9
adam_beta1 = 0.9
adam_beta2 = 0.999                                              #Beta values are the exponential decay rates for the moment estimates.
optimizer_using = 'ADAM'                                        #Options: 'ADAM', 'RMSprop'
#HyperParameters section ends here.

smote = SMOTE()
x_resample, y_resample = smote.fit_resample(trainingData, trainingData_actualResult)
y_resample = y_resample.reshape(-1, 1)

SCALED_training_data = np.hstack((x_resample, y_resample))

learning_rates = []
loss_value_list = []
confusion_matrices = []
regression_results = []
logit_results = []

while learningRate_weight <= 0.02:
    optimizer_class = classification.Optimizer(featureSize, optimizer_using, RMS_beta1, adam_beta1, adam_beta2)
    print(f"Training on Learning Rate: {learningRate_weight}\n")
    classificationNode = classification.model(featureSize, epoch, learningRate_weight, learningRate_bias, batchSize, POSITIVE_strata_size, 
                                    NEGATIVE_strata_size, momentum_coff, regularization_coff, decision_threshold, yeta_elastic_net, optimizer_class)
    classificationNode.training(np.array(trainingData))

    accuracy, precision, recall, f1, cm, LL_likelihood, LL_null = classificationNode.evaluation(evaulationData, evaluationData_actualResult)

    learning_rates.append(learningRate_weight)
    loss_value_list.append(classificationNode.loss_history)
    confusion_matrices.append(cm)

    LLR = 2 * (LL_likelihood - LL_null)
    p_value = 1 - chi2.sf(LLR, df=featureSize)
    R_squ = 1 - (LL_likelihood / LL_null)

    logit_results.append([
        f"Current function value: {classificationNode.loss_history[-1]:.6f}",
        f"Learning Rate: {learningRate_weight}", "Target Variable: diagnosis_M",
        f"No. Observations: {datasetSize}", "Model: Logit",
        f"Df Residuals: {datasetSize + featureSize}", "Method: MLE",
        f"Pseudo R-squ.: {R_squ}", f"Log-Likelihood: {LL_likelihood}.",
        f"LL-Null: {LL_null}.", f"LLR p-value: {p_value}",
        f"Momentum Coeff.: {momentum_coff}.", f"Regularization Coeff: {regularization_coff}",
        f"Yeta Elastic Net: {yeta_elastic_net}", #f"Batch Size: {batchSize}",
        f"Accuracy: {accuracy}.", f"Precision: {precision}",
        f"Recall: {recall}.", f"F1-Score: {f1}"
    ])

    del optimizer_class
    del classificationNode
    print("Training completed on Learning Rate: ", learningRate_weight)
    learningRate_weight = learningRate_weight + 0.001

reportNode = model_report_generator.binary_classification_report()
print("\n\nAll training completed.\n Generating Report.")
reportNode.generate_pdf(len(learning_rates), "StrokeRisk_Report_STRATIFIED_RMS.pdf", learning_rates, loss_value_list, confusion_matrices, logit_results, model_name)
print("\nREPORT GENERATED....")

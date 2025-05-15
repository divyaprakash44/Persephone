# Copyright 2025 Divya Prakash Singh
# Licensed under the Apache License, Version 2.0
# See the LICENSE file for more details.

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_curve, auc
import seaborn as sns
import numpy as np
import pandas as pd
import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.platypus import Table, TableStyle, Image
from reportlab.lib import colors

save_location = "C:/AI-ML/projects/reports/"

class binary_classification_report:
    def generate_loss_plot(self, c, x, y, learning_rate, loss_values):
        buffer = io.BytesIO()
        plt.figure(figsize=(7, 4))  # Increased figure size for better readability
        plt.plot(range(len(loss_values)), loss_values, marker='o', linestyle='-', color='b', linewidth=2, markersize=5)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(f"Loss Curve (LR={learning_rate})", fontsize=14)
        plt.grid(True, linestyle="--", linewidth=0.5)

        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()

        buffer.seek(0)
        img = Image(buffer, width=300, height=200)  # Increased size in PDF
        img.drawOn(c, x, y)

    def generate_confusion_matrix(self, c, x, y, cm):
        buffer = io.BytesIO()
        plt.figure(figsize=(4.5, 4))  # Increased figure size
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=1, linecolor='black', annot_kws={"size": 14})
        plt.xlabel("Predicted", fontsize=12)
        plt.ylabel("Actual", fontsize=12)
        plt.title("Confusion Matrix", fontsize=14)

        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()

        buffer.seek(0)
        img = Image(buffer, width=250, height=200)  # Increased size in PDF
        img.drawOn(c, x, y)

    def generate_pdf(self, no_page, output_filename, learning_rates, loss_values_list, confusion_matrices, logit_results, model_name):
        output_filename = os.path.join(save_location, output_filename)  # Save in the reports folder
        c = canvas.Canvas(output_filename, pagesize=letter)
        width, height = letter

        for i in range(no_page):
            y_position = height - 80
            # Add Logo at Top Right
            try:
                c.drawImage("C:/AI-ML/projects/util/logo.png", width - 150, height - 120, width=150, height=150, preserveAspectRatio=True, mask='auto')
            except:
                print("Warning: Logo file not found or could not be loaded.")

            c.setFont("Helvetica-Bold", 16)
            c.drawString(150, y_position, f"{model_name} Report")
            y_position -= 30
            c.setFont("Helvetica", 12)
            c.drawString(50, y_position, f"Learning Rate: {learning_rates[i]}")
            y_position -= 20
            
            # Plot and add Loss Graph directly
            self.generate_loss_plot(c, 50, y_position - 200, learning_rates[i], loss_values_list[i])
            
            # Plot and add Confusion Matrix directly
            self.generate_confusion_matrix(c, 300, y_position - 200, confusion_matrices[i])
            y_position -= 210
            
            # Logit Regression Results
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_position, "Logit Regression Results")
            y_position -= 30
            c.setFont("Helvetica", 10)
            for line in logit_results[i]:
                c.drawString(50, y_position, line)
                y_position -= 15
            
            # Logistic Regression-style Output Table
            """data = [["", "coef", "std err", "z", "P>|z|", "[95% Conf. Int]"]]
            for row in regression_results[i]:
                data.append(row)
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            table.wrapOn(c, width, height)
            table.drawOn(c, 50, y_position - 100)"""

            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y_position, "Glossary")
            y_position -= 30
            c.setFont("Helvetica", 10)
            glossary_items = [
                "1. Df residual: Degrees of freedom (Df) measures how much free information is left after estimating model parameters.",
                "2. Pseudo R-square: Measures how well the model explains the variance.",
                "3. Log-Likelihood (LL): Likelihood of observed data given model parameters (higher is better).",
                "4. LL-Null: Log-Likelihood when model has no features—only the intercept.",
                "5. LLR p-value (Likelihood Ratio Test): Checks if adding features significantly improves the model using chi-square test."
            ]

            for item in glossary_items:
                c.drawString(50, y_position, item)
                y_position -= 15

            c.showPage()
        
        c.save()


class multi_class_classifier():
    def generate_multiclass_confusion_matrix(self, c, x, y, y_true, y_pred, class_labels):
        cm = confusion_matrix(y_true, y_pred)
        buffer = io.BytesIO()
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, linewidths=1, linecolor='black')
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.xticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=45)
        plt.yticks(ticks=np.arange(len(class_labels)), labels=class_labels, rotation=0)
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        img = Image(buffer, width=250, height=200)
        img.drawOn(c, x, y)

    def generate_precision_recall_bar(self, c, x, y, y_true, y_pred, class_labels):
        precision, recall, _, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        buffer = io.BytesIO()
        plt.figure(figsize=(5, 4))
        x_labels = np.arange(len(class_labels))
        plt.bar(x_labels - 0.2, precision, width=0.4, label='Precision', color='blue')
        plt.bar(x_labels + 0.2, recall, width=0.4, label='Recall', color='green')
        plt.xlabel("Classes")
        plt.ylabel("Score")
        plt.xticks(ticks=x_labels, labels=class_labels, rotation=45)
        plt.title("Precision & Recall per Class")
        plt.legend()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        img = Image(buffer, width=250, height=200)
        img.drawOn(c, x, y)

    def generate_multiclass_roc(self, c, x, y, y_true_one_hot, y_pred_prob, class_labels):
        buffer = io.BytesIO()
        plt.figure(figsize=(5, 4))
        for i in range(y_true_one_hot.shape[1]):
            fpr, tpr, _ = roc_curve(y_true_one_hot[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'Class {class_labels[i]} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve for Multiclass")
        plt.legend()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()
        buffer.seek(0)
        img = Image(buffer, width=250, height=200)
        img.drawOn(c, x, y)
    
    def generate_loss_plot(self, c, x, y, learning_rate, loss_values):
        buffer = io.BytesIO()
        plt.figure(figsize=(7, 4))  # Increased figure size for better readability
        plt.plot(range(len(loss_values)), loss_values, marker='o', linestyle='-', color='b', linewidth=2, markersize=5)
        plt.xlabel("Epochs", fontsize=12)
        plt.ylabel("Loss", fontsize=12)
        plt.title(f"Loss Curve (LR={learning_rate})", fontsize=14)
        plt.grid(True, linestyle="--", linewidth=0.5)

        plt.savefig(buffer, format='png', bbox_inches='tight')
        plt.close()

        buffer.seek(0)
        img = Image(buffer, width=300, height=200)  # Increased size in PDF
        img.drawOn(c, x, y)

    def generate_multiclass_classification_report(self, no_page, y_true, y_pred, y_true_one_hot, y_pred_prob, class_labels,
                                                  output_filename, learning_rates, loss_values_list, logit_results, 
                                                  model_name):
        
        output_filename = os.path.join(save_location, output_filename)  # Save in the reports folder
        c = canvas.Canvas(output_filename, pagesize=letter)
        width, height = letter

        for i in range(no_page):
            y_position = height - 80
            # Add Logo at Top Right
            try:
                c.drawImage("C:/AI-ML/projects/util/logo.png", width - 150, height - 120, width=150, height=150, preserveAspectRatio=True, mask='auto')
            except:
                print("Warning: Logo file not found or could not be loaded.")

            c.setFont("Helvetica-Bold", 16)
            c.drawString(150, y_position, f"{model_name} Report")
            y_position -= 30
            c.setFont("Helvetica", 12)
            c.drawString(50, y_position, f"Learning Rate: {learning_rates}")
            y_position -= 20
            
            # Plot and add Loss Graph directly
            self.generate_loss_plot(c, 10, y_position - 200, learning_rates, loss_values_list)
            
            # Plot and add Confusion Matrix directly
            self.generate_multiclass_confusion_matrix(c, 300, y_position - 200, y_true, y_pred, class_labels)
            y_position -= 420

            self.generate_precision_recall_bar(c, 10, y_position, y_true, y_pred, class_labels)

            self.generate_multiclass_roc(c, 300, y_position, y_true_one_hot, y_pred_prob, class_labels)
            y_position -= 20
            
            # Logit Regression Results
            c.setFont("Helvetica-Bold", 12)
            c.drawString(10, y_position, "Logit Regression Results")
            y_position -= 30
            c.setFont("Helvetica", 10)
            for line in logit_results[i]:
                c.drawString(50, y_position, line)
                y_position -= 15
            
            # Logistic Regression-style Output Table
            """data = [["", "coef", "std err", "z", "P>|z|", "[95% Conf. Int]"]]
            for row in regression_results[i]:
                data.append(row)
            
            table = Table(data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 5),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            table.wrapOn(c, width, height)
            table.drawOn(c, 50, y_position - 100)"""

            c.showPage()
        
        c.save()

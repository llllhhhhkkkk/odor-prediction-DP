import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import re


directory = './ensemble_cv_exp/ensemble'

# methods = ['DGN', 'GINE', 'GatedGCN', 'GGS-NNs', 'GMM', 'MPNN', 'PNA', 'R-GCN', 'EdgeGAT', 'CFNN']
methods = ['Mean', 'Weight Mean', 'Max Min', 'Booting Single Model', 'Booting Single Model(no fold)',
           'Booting Different Model(no fold)', 'Stacking Logistic', 'Stacking Nerual Network',
           'Stacking Nerual Network (k fold)', 'Dynamic', 'Feature Ensemble (simple)', 'Feature Ensemble (mean)',
           'Feature Ensemble (nerual network)', 'Feature Ensemble (attention)', 'Feature Ensemble (self attention)']



precision_values = []
recall_values = []
f1_values = []


for file in os.listdir(directory):
    if file.endswith('.txt'):
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as file:
            data = file.read()


        data = re.sub(r'\[([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\]', r'[\1, \2, \3]', data)

        cv_mean_micro_averages = ast.literal_eval(data.split('\n')[3].split('= ')[1])


        precision_values.append(cv_mean_micro_averages[0])
        recall_values.append(cv_mean_micro_averages[1])
        f1_values.append(cv_mean_micro_averages[2])


plt.figure(figsize=(10, 6))
bars = plt.bar(methods, precision_values, color='#AABcDB')
# bars = plt.bar(methods, precision_values, color='#A8D8B8')
plt.xlabel('Methods')
plt.ylabel('Precision')
plt.title('Precision Values Across Different GNNs')
plt.xticks(rotation=60, fontsize=8)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('precision_chart.png')
plt.close()


plt.figure(figsize=(10, 6))
bars = plt.bar(methods, recall_values, color='#7698C3')
# bars = plt.bar(methods, recall_values, color='#80C6D2')
plt.xlabel('Methods')
plt.ylabel('Recall')
plt.title('Recall Values Across Different GNNs')
plt.xticks(rotation=60, fontsize=8)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('recall_chart.png')
plt.close()


plt.figure(figsize=(10, 6))
bars = plt.bar(methods, f1_values, color='#487DB2')
# bars = plt.bar(methods, f1_values, color='#57B1DB')
plt.xlabel('Methods')
plt.ylabel('F1-score')
plt.title('F1-score Values Across Different GNNs')
plt.xticks(rotation=60, fontsize=8)

for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval, f'{yval:.3f}', ha='center', va='bottom')
plt.tight_layout()
plt.savefig('f1_score_chart.png')
plt.close()

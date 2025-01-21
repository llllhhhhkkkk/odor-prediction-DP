import ast
import os
import matplotlib.pyplot as plt
import numpy as np
import re

# 文件夹路径
directory = 'open_pom_feature_ensemble/ensemble_cv_exp'
for file in os.listdir(directory):
    if file.endswith('.txt'):
        file_path = os.path.join(directory, file)
        with open(file_path, 'r') as file:
            data = file.read()


        data = re.sub(r'\[([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\]', r'[\1, \2, \3]', data)

        folds_results = ast.literal_eval(data.split('\n')[0].split('= ')[1])
        cv_mean_result = float(data.split('\n')[1].split('= ')[1])
        folds_micro_averages = ast.literal_eval(data.split('\n')[2].split('= ')[1])
        cv_mean_micro_averages = ast.literal_eval(data.split('\n')[3].split('= ')[1])

        base_path = os.path.splitext(file_path)[0]
        line_chart_path = f"{base_path}_line_chart.png"
        bar_chart_path = f"{base_path}_bar_chart.png"
        # box_plot_path = f"{base_path}_box_plot.png"

        # Fold Results Line Chart
        plt.figure(figsize=(10, 6))
        folds = list(range(1, 6)) + ["CV Mean"]
        results = folds_results + [cv_mean_result]
        plt.plot(folds, results, linestyle='-', color='#219ebc')
        for i, txt in enumerate(results):
            plt.annotate(f'{txt:.3f}', (i, results[i]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.title('Fold Results')
        plt.xlabel('Fold')
        plt.ylabel('Result')
        plt.grid(True)
        plt.savefig(line_chart_path)
        plt.close()

        # Micro Averages Bar Chart
        micro_averages = np.array(folds_micro_averages + [cv_mean_micro_averages])
        labels = ['Precision', 'Recall', 'F1-score']
        x = np.arange(len(labels))

        fig, ax = plt.subplots(figsize=(12, 8))
        width = 0.15  # the width of the bars


        colors = ['#1f77b4', '#6baed6', '#9edae5', '#c5b0d5', '#9467bd', '#8c564b']

        for i in range(6):
            ax.bar(x + (i-2.5)*width, micro_averages[i], width, label=f'Fold {i+1}' if i < 5 else 'CV Mean', color=colors[i])

        ax.set_xlabel('Metrics')
        ax.set_ylabel('Scores')
        ax.set_title('Micro Averages for Each Fold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1.0)  # Set y-axis limit to 1.0
        ax.legend()
        plt.savefig(bar_chart_path)
        plt.close()

        # Box Plot for Fold Results
        # plt.figure(figsize=(10, 6))
        # plt.boxplot(folds_results, vert=True, patch_artist=True, labels=['Fold Results'])
        # plt.title('Box Plot for Fold Results')
        # plt.ylabel('Result')
        # plt.grid(True)
        # plt.savefig(box_plot_path)
        # plt.close()

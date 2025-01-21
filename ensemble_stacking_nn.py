import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc as sklearn_auc
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset


folds = 5
n_models = 10
folds_results = []
micro_averages = []

# Initialize variables to calculate the mean ROC
mean_fpr = np.linspace(0, 1, 100)
tprs = []
mean_auc = 0.0
colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'purple']
for k in range(folds):
    model_predictions = []
    for i in range(n_models):
        predictions = np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/train_preds_{i + 1}.npy')
        model_predictions.append(predictions)
    model_predictions = np.array(model_predictions)
    train_stacked_features = model_predictions.transpose((1, 0, 2)).reshape(model_predictions.shape[1], -1)
    
    model_predictions_test = []
    for i in range(n_models):
        predictions = np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/test_preds_{i + 1}.npy')
        model_predictions_test.append(predictions)
    model_predictions_test = np.array(model_predictions_test)
    test_stacked_features = model_predictions_test.transpose((1, 0, 2)).reshape(model_predictions_test.shape[1], -1)

    train_dataset_y = np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/train_true_label.npy')
    
    test_dataset_y = np.load(f'./ensemble_cv_exp/stacking/result/fold_{k + 1}/test_true_label.npy')

    y = torch.tensor(train_dataset_y, dtype=torch.float32)
    X = np.hstack([pred.reshape(model_predictions[0].shape[0], 138) for pred in model_predictions])
    X = torch.tensor(X, dtype=torch.float32)
    
    y_test = torch.tensor(test_dataset_y, dtype=torch.float32)
    X_test = np.hstack([pred.reshape(model_predictions_test[0].shape[0], 138) for pred in model_predictions_test])
    X_test = torch.tensor(X_test, dtype=torch.float32)
    
    class SimpleNN(nn.Module):
        def __init__(self, input_size=X.shape[1], output_size=138, step_nums=1000, dropout_rate=0.5):
            super(SimpleNN, self).__init__()
            # self.normalization = nn.BatchNorm1d(input_size)
            feature_size = input_size
            self.NN = nn.ModuleList()
            while feature_size > output_size + step_nums:
                self.NN.append(nn.Linear(feature_size, feature_size - step_nums))
                # self.NN.append(nn.BatchNorm1d(feature_size - step_nums))
                self.NN.append(nn.ReLU())
                # self.NN.append(nn.Dropout(dropout_rate))
                feature_size -= step_nums
            self.NN.append(nn.Linear(feature_size, output_size))
            print(self.NN)
        def forward(self, x):
            # x = self.normalization(x)
            for layer in self.NN:
                x = layer(x)
            return x
        
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # 创建DataLoader
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    train_losses = []
    val_losses = []
    model = SimpleNN().to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=60, gamma=0.01)
    
    for epoch in range(400):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu'), y_batch.to(
                'cuda' if torch.cuda.is_available() else 'cpu')
            optimizer.zero_grad()
            outputs = model(X_batch)
            train_loss = criterion(outputs, y_batch)
            train_loss.backward()
            optimizer.step()
    
            epoch_train_loss += train_loss.item()
        train_losses.append(epoch_train_loss / len(train_loader))
    
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu'), y_batch.to(
                    'cuda' if torch.cuda.is_available() else 'cpu')
                val_outputs = model(X_batch)
                val_loss = criterion(val_outputs, y_batch)
                epoch_val_loss += val_loss.item()
        val_losses.append(epoch_val_loss / len(val_loader))
    
        # scheduler.step()
    
        print(f'Epoch {epoch}: Training Loss {train_loss.item()}, Validation Loss {val_loss.item()}')
    
    epochs = range(0, len(train_losses))
    
    # plt.plot(epochs, train_losses, label='Training Loss')
    # plt.plot(epochs, val_losses, label='Validation Loss')
    # plt.xlabel('Epochs')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss Curves')
    # plt.legend()
    # plt.show()
    
    with torch.no_grad():
        model.eval()
        X_test = X_test.to('cuda' if torch.cuda.is_available() else 'cpu')
        y_test = y_test.to('cuda' if torch.cuda.is_available() else 'cpu')
        predictions = model(X_test).cpu().detach().numpy()
        y_test_np = y_test.cpu().detach().numpy()

        threshold = 0.65
        y_pred = (predictions >= threshold).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(y_test_np, y_pred, average=None,
                                                                   zero_division=0)

        # for i, (p, r, f) in enumerate(zip(precision, recall, f1)):
        #     print(f'Label {i}: Precision = {p:.2f}, Recall = {r:.2f}, F1-score = {f:.2f}')


        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test_np, y_pred,
                                                                                     average='macro',
                                                                                     zero_division=0)
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_test_np, y_pred,
                                                                                     average='micro',
                                                                                     zero_division=0)

        print(
            f'Macro-average: Precision = {precision_macro:.3f}, Recall = {recall_macro:.3f}, F1-score = {f1_macro:.3f}')
        print(
            f'Micro-average: Precision = {precision_micro:.3f}, Recall = {recall_micro:.3f}, F1-score = {f1_micro:.3f}')
        mi_average = [precision_micro, recall_micro, f1_micro]
        fold_result = roc_auc_score(y_test_np, predictions, average="macro")
        folds_results.append(fold_result)
        micro_averages.append(mi_average)

        # Calculate ROC curve and AUC for each class
        fpr, tpr, _ = roc_curve(y_test_np.ravel(), predictions.ravel())
        roc_auc_value = sklearn_auc(fpr, tpr)
        mean_auc += roc_auc_value
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
    
    torch.cuda.empty_cache()

cv_mean_result = np.mean(folds_results)
cv_mean_micro_averages = np.mean(micro_averages, axis=0)

mean_auc /= folds
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_roc_auc = sklearn_auc(mean_fpr, mean_tpr)

# Plot the mean ROC curve
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f)' % mean_roc_auc, lw=2, alpha=.8)
# Plot the ROC curve for each fold
# for i, color in enumerate(colors[:fold]):
#     plt.plot(mean_fpr, tprs[i], lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i+1, folds_results[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic for All Folds')
plt.legend(loc="lower right")
plt.savefig("./ensemble_cv_exp/AUC-ROC/ensemble_stacking_nn.png")

with open("./ensemble_cv_exp/final_score_stacking_nn.txt", 'w+') as f:
    f.write(f"folds_results = {folds_results}\n")
    f.write(f"cv_mean_result = {cv_mean_result}\n")
    f.write(f"folds_micro_averages = {micro_averages}\n")
    f.write(f"cv_mean_micro_averages = {cv_mean_micro_averages}\n")
round(cv_mean_result, 4)
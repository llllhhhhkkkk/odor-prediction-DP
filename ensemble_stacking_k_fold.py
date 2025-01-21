import pandas as pd
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, roc_curve, auc as sklearn_auc
import torch.optim as optim
import matplotlib.pyplot as plt
from torch import nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt


class SimpleNN(nn.Module):
    def __init__(self, input_size=1380, output_size=138, step_nums=138, dropout_rate=0.5):
        super(SimpleNN, self).__init__()
        feature_size = input_size
        self.NN = nn.ModuleList()
        while feature_size > output_size + step_nums:
            self.NN.append(nn.Linear(feature_size, feature_size - step_nums))
            # self.NN.append(nn.BatchNorm1d(feature_size - step_nums))
            self.NN.append(nn.LeakyReLU())
            # self.NN.append(nn.Dropout(dropout_rate))
            feature_size -= step_nums
        self.NN.append(nn.Linear(feature_size, output_size))
        print(self.NN)

    def forward(self, x):
        for layer in self.NN:
            x = layer(x)
        return x


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=True, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')

        pt = torch.exp(-BCE_loss)  # Prevents nans when probability 0
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss()

    def forward(self, inputs, targets):
        loss_bce = self.bce_loss(inputs, targets)
        loss_focal = self.focal_loss(inputs, targets)
        return 0.5 * loss_bce + 0.5 * loss_focal


def apply_z_score_standardization(data):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    standardized_data = (data - mean) / (std + 1e-6)
    return standardized_data, mean, std


model_list = [0, 1, 2, 3, 4]
fold = 5
n_models = 10
folds_results = []
micro_averages = []

mean_fpr = np.linspace(0, 1, 100)
tprs = []
mean_auc = 0.0
colors = ['aqua', 'darkorange', 'cornflowerblue', 'darkgreen', 'purple']

for k in range(fold):
    all_means = []
    all_stds = []
    new_data_train = []
    for j in range(len(model_list)):
        new_data_models = []
        for i in [0, 1, 2, 3, 4]:
            data = pd.read_excel(f'./ensemble_cv_exp/fold_{k + 1}/model{j + 1}/result_vail/pred_vail_{i + 1}.xlsx').values
            standardized_data, mean, std = apply_z_score_standardization(data)
            new_data_models.append(standardized_data)
            if i == 0:
                all_means.append(mean)
                all_stds.append(std)
        new_data_train.append(np.concatenate(new_data_models, axis=0))
    new_data_train = torch.tensor(np.hstack(new_data_train), dtype=torch.float32)
    print(new_data_train.shape)
    
    new_data_models_true = []
    for i in [0, 1, 2, 3, 4]:
        new_data_models_true.append(
            pd.read_excel(f'./ensemble_cv_exp/fold_{k + 1}/model1/result_vail/true_vail_{i + 1}.xlsx').values)
    new_data_train_true = torch.tensor(np.concatenate(new_data_models_true, axis=0), dtype=torch.float32)
    print(new_data_train_true.shape)
    
    new_test_data = []
    for j in range(len(model_list)):
        test_data = []
        for i in [0, 1, 2, 3, 4]:
            data = pd.read_excel(f'./ensemble_cv_exp/fold_{k + 1}/model{j + 1}/result_test/pred_test_{i + 1}.xlsx').values
            standardized_data = (data - all_means[j]) / (all_stds[j] + 1e-6)
            test_data.append(standardized_data)
        test_data_mean = np.mean(test_data, axis=0)
        new_test_data.append(test_data_mean)
    new_test_data = torch.tensor(np.hstack(new_test_data), dtype=torch.float32)
    print(new_test_data.shape)
    
    new_test_data_true = torch.tensor(pd.read_excel(f'./ensemble_cv_exp/fold_{k + 1}/model1/result_test/true_test.xlsx').values,
                                      dtype=torch.float32)
    print(new_test_data_true.shape)
    
    new_data_train_np = new_data_train.numpy()
    new_data_train_true_np = new_data_train_true.numpy()
    new_test_data_np = new_test_data.numpy()
    new_test_data_true_np = new_test_data_true.numpy()
    
    X_train, X_val, y_train, y_val = train_test_split(new_data_train, new_data_train_true, test_size=0.2, random_state=42)
    
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    # 创建DataLoader
    batch_size = 1024
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    train_losses = []
    val_losses = []
    model = SimpleNN(input_size=X_train.shape[1], step_nums=276)  # .to('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = CombinedLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=200, gamma=0.1)
    for epoch in range(50):
        model.train()
        epoch_train_loss = 0
        for X_batch, y_batch in train_loader:
            # X_batch, y_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu'), y_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
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
                # X_batch, y_batch = X_batch.to('cuda' if torch.cuda.is_available() else 'cpu'), y_batch.to('cuda' if torch.cuda.is_available() else 'cpu')
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
        # X_test = X_test.to('cuda' if torch.cuda.is_available() else 'cpu')
        # y_test = y_test.to('cuda' if torch.cuda.is_available() else 'cpu')
        predictions = model(new_test_data).numpy()
        y_test_np = new_test_data_true.numpy()


        threshold = 0.65
        y_pred = (predictions >= threshold).astype(int)
        # 计算每个标签的precision, recall, f1-score
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
        fpr, tpr, _ = roc_curve(y_test_np.y.ravel(), predictions.ravel())
        roc_auc_value = sklearn_auc(fpr, tpr)
        mean_auc += roc_auc_value
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0

    torch.cuda.empty_cache()

cv_mean_result = np.mean(folds_results)
cv_mean_micro_averages = np.mean(micro_averages, axis=0)

mean_auc /= fold
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
plt.savefig("./ensemble_cv_exp/AUC-ROC/ensemble_stacking_k_fold.png")

with open("./ensemble_cv_exp/final_score_stacking_k_fold.txt", 'w+') as f:
    f.write(f"folds_results = {folds_results}\n")
    f.write(f"cv_mean_result = {cv_mean_result}\n")
    f.write(f"folds_micro_averages = {micro_averages}\n")
    f.write(f"cv_mean_micro_averages = {cv_mean_micro_averages}\n")
round(cv_mean_result, 4)

import pandas as pd
import numpy as np
import random
import optuna
import time
import torch
from torch import nn, optim
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import sys
import argparse
from tqdm import tqdm

sys.path.append('Utils/')
import global_config
import DataProcessing as DP

"""
This code is obtained from: 

Giuseppina Andresini, Feargus Pendlebury, Fabio Pierazzi, Corrado Loglisci, Annalisa Appice, and Lorenzo Cavallaro "INSOMNIA: Towards Concept-Drift Robustness in Network Intrusion Detection", AISec 2021. 



"""
parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default="CICIDS2017", choices=["CICIDS2017", "CICIDS2018"])

args = parser.parse_args()

print(' Hyperparameter Search for the DNN softmax classifier architecture')

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
device


x_train, x_valid, x_test, x_drift, x_non_drift, y_train, y_valid, y_test, y_drift, y_non_drift =  DP.data_processing(args.dataset_name)
num_classes = len(np.unique(y_train))

class Net(nn.Module):
    def __init__(self, input_shape, params):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_shape, params['neurons1'])
        self.dropout1 = nn.Dropout(params['dropout1'])
        self.fc2 = nn.Linear(params['neurons1'], params['neurons2'])
        self.dropout2 = nn.Dropout(params['dropout2'])
        self.fc3 = nn.Linear(params['neurons2'], params['neurons3'])
        self.fc4 = nn.Linear(params['neurons3'], global_config.n_class)
        self.activation = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.dropout1(x)
        x = self.activation(self.fc2(x))
        x = self.dropout2(x)
        x = self.activation(self.fc3(x))
        x = self.softmax(self.fc4(x))
        return x
def NN(x_train, y_train, params):
    input_shape = x_train.shape[1]
    print(input_shape)
    class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    class_weights = torch.FloatTensor(class_weights).to(device)

    model = Net(input_shape, params).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    x_train_tensor = torch.tensor(x_train, dtype=torch.float)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    x_validation_tensor = torch.tensor(x_valid, dtype=torch.float)
    y_validation_tensor = torch.tensor(y_valid, dtype=torch.long)
    train_data = torch.utils.data.TensorDataset(x_train_tensor, y_train_tensor)
    val_data = torch.utils.data.TensorDataset(x_validation_tensor, y_validation_tensor)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=params['batch'], shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=params['batch'], shuffle=True, num_workers=4)
    
    best_loss = np.inf
    best_epoch = -1
    patience = 10

    for epoch in tqdm(range(150)):
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                val_outputs = model(inputs)
                val_loss += criterion(val_outputs, targets).item()
            val_loss /= len(val_loader)
            #print(f' Epoch: {epoch} -- Val Loss: {val_loss}')
            if val_loss < best_loss:
                best_loss = val_loss
                best_epoch = epoch
                best_model = model.state_dict()
            elif epoch - best_epoch > patience:
                break

    model.load_state_dict(best_model)
    torch.cuda.empty_cache()
    return model, best_loss


def fit_and_score(trial):
    params = {
        'batch': trial.suggest_categorical('batch', [32, 64, 128, 256, 512]),
        'dropout1': trial.suggest_float('dropout1', 0, 1),
        'dropout2': trial.suggest_float('dropout2', 0, 1),
        'learning_rate': trial.suggest_float('learning_rate', 0.0001, 0.01, log=True),
        'neurons1': trial.suggest_categorical('neurons1', [128, 256, 512]),
        'neurons2': trial.suggest_categorical('neurons2', [64, 128, 256]),
        'neurons3': trial.suggest_categorical('neurons3', [32, 64, 128]),
    }
    
    try:
        model, val = NN(global_config.train_X, global_config.train_Y, params)

        with torch.no_grad():
            x_test_tensor = torch.tensor(global_config.test_X, dtype=torch.long).to(device)
            y_test_tensor = torch.tensor(global_config.test_Y, dtype=torch.long).to(device)
            x_test_tensor = x_test_tensor.float()
            outputs = model(x_test_tensor)

            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == y_test_tensor).sum().item()
            accuracy = correct / len(y_test_tensor)
            
            if accuracy > global_config.best_accuracy:
                global_config.best_model = model
                global_config.best_accuracy = accuracy
        return val  

    except RuntimeError as e:
        if "out of memory" in str(e):
            torch.cuda.empty_cache()
            print(f"Out of memory error at trial {trial.number}. Skipping.")
            raise optuna.TrialPruned()
        else:
            raise e


def hypersearch(train_X, train_Y, test_X, test_Y, modelPath, n_class):
    reset_global_variables(train_X, train_Y, test_X, test_Y)
    global_config.n_class=n_class
    global_config.test_path = modelPath
    sampler = optuna.samplers.TPESampler(seed=seed)  
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(fit_and_score, n_trials=20)
    best_model = study.best_params
    print(study.best_trial)
    return best_model

def reset_global_variables(train_X, train_Y, test_X, test_Y):
    global_config.train_X = train_X
    global_config.train_Y = train_Y
    global_config.test_X = test_X
    global_config.test_Y = test_Y

    global_config.best_score = 0
    global_config.best_scoreTest = 0
    global_config.best_accuracy=0
    global_config.best_model = None
    global_config.best_model_test = None
    global_config.best_time = 0

best_model = hypersearch(x_train, y_train, x_valid, y_valid, args.dataset_name, num_classes)

print(f'The best model for the {args.dataset_name} dataset is')
print(best_model)


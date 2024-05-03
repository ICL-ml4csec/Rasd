import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
import CentroidDefinition as CD
import DataProcessing as DP
import Models
import CLlosses
import Metrics
import os
import time
import copy

seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"
device


def validation_loss(model, dataset, param, distance_loss="Rasd"):
    losses = 0
    for data, target in dataset:
        projection = model(data)
        if distance_loss=="Rasd":
            loss = CLlosses.Rasd_loss(projection, target.cuda(), ensure_tensor(param))
        elif distance_loss=="LSL":
            loss = CLlosses.LSL_loss(projection, target.cuda(), ensure_tensor(param))
        losses += loss.item()
    return losses

def print_lr(optimizer, print_screen=True,epoch = -1):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        if print_screen == True:
            print(f'learning rate : {lr:.3f}')
    return lr


def get_lr(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))


def ensure_tensor(data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float).to(device)
    else:
        data = data.to(device)
    return data

def train_Rasd(train_loader, model, optimizer, scheduler, epoch, loss_param=None, distance_loss="Rasd"):
    model.train()
    train_bar = train_loader
    lr = print_lr(optimizer, epoch)
    total_min_batch_loss = 0
    num_class = 8
    for idx, (img, target) in enumerate(train_bar):
        optimizer.zero_grad()
        labels = target.cuda()
        
        param = ensure_tensor(loss_param)
        rep= model(img)
        if distance_loss=="Rasd":
            Loss = CLlosses.Rasd_loss(rep, labels, param)
        elif distance_loss=="LSL":
            Loss = CLlosses.LSL_loss(rep, labels, param)
        total_min_batch_loss += Loss.item()
        total_loss =  Loss
        total_loss.backward()
        optimizer.step()
        scheduler.step()
    return model, total_min_batch_loss


def Get_Rasd_Trained_Model(model, x_train, y_train, x_valid, y_valid, syst, dataset_name, param=None, distance_loss="Rasd"):
    
    train_dataset, valid_dataset = DP.Rasd_loader(x_train, y_train), DP.Rasd_loader(x_valid, y_valid, batch_size=512)

    learning_rate = 0.0001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    total_step = 250 * len(train_dataset)
    scheduler = CosineAnnealingLR(optimizer, T_max= total_step, eta_min=learning_rate * 5e-4) 
    assert torch.cuda.is_available()
    cudnn.benchmark = True
    model = nn.DataParallel(model).cuda()
    best_loss = float('inf')
    for epoch in tqdm(range(1, 250+1)):
        model.train()
        model, total_min_batch_loss = train_Rasd(train_dataset, model, optimizer, scheduler, epoch, loss_param=param, distance_loss=distance_loss)
        with torch.no_grad():
            total_epoch_score = validation_loss(model, valid_dataset, param, distance_loss=distance_loss)
            if total_epoch_score < best_loss:
                best_loss = total_epoch_score
                torch.save(model.module.state_dict(), f'Models/{distance_loss}/{dataset_name}/{syst}.pth')
    print(f'Val Loss {best_loss}')
    return model


def find_best_model(x_valid, y_valid, model_dir, model_class, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    x_val = torch.tensor(x_valid, dtype=torch.float32).to(device)
    y_val = torch.tensor(y_valid, dtype=torch.long)  
    best_score = 0.0
    best_model_name = ''
    
    for filename in os.listdir(model_dir):
        if (filename.endswith('.pt') or filename.endswith('.pth')):
            model_path = os.path.join(model_dir, filename)
            model = model_class(x_valid.shape[1]).to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            
            val_latent = model(x_val).detach().cpu().numpy()
            print(f'Evaluating model: {model_path}')
            score = Metrics.evaluate_clustering(val_latent, y_valid.astype(int))
            
            if score > best_score:
                best_score = score
                best_model_name = model_path
    print('Best Model:', best_model_name)
    print('Best Score:', best_score)
    return best_model_name, best_score

def GA(x_train, latent_size, num_classes, y_train, x_valid, y_valid, ds_name):
    for i in range(1,11):
        model = Models.Encoder(x_train.shape[1]) 
        classes_centroids = CD.GA_centroids(model, x_train, latent_size, num_classes, i) 
        trained_model = Get_Rasd_Trained_Model(model, x_train, y_train, x_valid, y_valid, i, ds_name, param=classes_centroids)
    return



def Rasd_HP(x_train, y_train, x_valid, y_valid, latent_size, num_classes, ds_name, mode):
    if mode == "train-new":
        GA(x_train, latent_size, num_classes, y_train, x_valid, y_valid, ds_name)
        print('Done: Training For Multiple Centroids')
    model_dir = f'Models/Rasd/{ds_name}/'
    best_model_name, _ = find_best_model(x_valid, y_valid, model_dir, Models.Encoder, device=None)
    print('Done: HyperParameter Search For Rasd')
    print(best_model_name)
    model_class = Models.Encoder(x_valid.shape[1])
    model_class.load_state_dict(torch.load(f'{best_model_name}'))
    return model_class


def LSL_HP(x_train, y_train, x_valid, y_valid, latent_size, num_classes, ds_name, distance_loss, mode):
    def train(model, classes_centroids, distnace_loss, ds_name=ds_name, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid):
        trained_model = Get_Rasd_Trained_Model(model, x_train, y_train, x_valid, y_valid, classes_centroids, ds_name, param=classes_centroids, distance_loss=distance_loss)
        return
    alpha_list = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]
    if mode == "train-new":
        for i in alpha_list:
            model = Models.Encoder(x_valid.shape[1]) 
            train(model, i, distance_loss)
        print('Done: Training For Multiple Alphas')
    model_dir = f'Models/{distance_loss}/{ds_name}/'
    best_model_name, _ = find_best_model(x_valid, y_valid, model_dir, Models.Encoder, device=None)
    print('Done: HyperParameter Search For LSL')
    print(best_model_name)
    model_class = Models.Encoder(x_valid.shape[1])
    model_class.load_state_dict(torch.load(f'{best_model_name}'))
    return model_class


def train_CADE(train_loader, model, optimizer, scheduler, epoch, margin, lambda_1):
    model.train()
    train_bar = train_loader
    total_epoch_loss = 0
    lr = print_lr(optimizer, epoch)
    for batch_idx, (x1, y1, x2, y2) in enumerate(train_bar):
        x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
        optimizer.zero_grad()
        contrastive_loss, recon_loss = CLlosses.CADE_loss(model,x1,x2, y1,y2, margin)
        loss = lambda_1 * contrastive_loss + recon_loss
        total_epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    return model, total_epoch_loss

def test_CADE(train_loader, model, margin, lambda_1):
    model.eval()
    train_bar = train_loader
    total_epoch_loss = 0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch_idx, (x1, y1, x2, y2) in enumerate(train_bar):
            x1, y1, x2, y2 = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
            outputs1 = model(x1)
            outputs2 = model(x2) # model, x1, x2, y1, y2, margin
            contrastive_loss, recon_loss = CLlosses.CADE_loss(model,x1,x2, y1,y2, margin)
            loss = lambda_1 * contrastive_loss + recon_loss
            total_epoch_loss += loss.item()
    return total_epoch_loss


def CADE_model(ds_name, x_train, y_train, batch_size, similar_samples_ratio, margin_rate, lambda_value, valid_loader):
    learning_rate = 0.0001
    model = Models.Autoencoder(x_train.shape[1]).to(device)
    optimizer_fn = torch.optim.Adam
    optimizer = optimizer_fn(model.parameters(), lr=learning_rate)
    X = torch.tensor(x_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.long)
    dataloader = DP.CADE_dataloader(X, y, batch_size=batch_size, similar_samples_ratio=similar_samples_ratio)
    total_step = 250 * len(dataloader)
    scheduler = CosineAnnealingLR(optimizer, T_max= total_step, eta_min=learning_rate * 5e-4)
    model = nn.DataParallel(model)
    best_loss = float('inf')
    for epoch in tqdm(range(1, 250+1)):
        model.train()
        model, _ = train_CADE(dataloader, model, optimizer, scheduler, epoch, margin_rate, lambda_value)
        with torch.no_grad():
            total_epoch_score = test_CADE(valid_loader, model, margin_rate, lambda_value)
            if total_epoch_score < best_loss:
                best_loss = total_epoch_score
                torch.save(model.module.state_dict(), f'Models/CADE/{ds_name}/CADE-m{margin_rate}-l{lambda_value}.pth')               
    return best_loss


def CADE_HP(x_train, y_train, x_valid, y_valid, ds_name, mode):
    lmbda_list = [1.0, 0.1, 0.01, 0.001]
    margin_list = [1.0, 5.0, 10.0, 15.0, 20.0]
    batch_size = 1024
    similar_samples_ratio = 0.25
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == "train-new":
        x_val = torch.tensor(x_valid, dtype=torch.float32)
        y_val = torch.tensor(y_valid, dtype=torch.long)
        val_dataloader = DP.CADE_dataloader(x_val, y_val, batch_size=batch_size,
                                    similar_samples_ratio=similar_samples_ratio)
        for lmbda in lmbda_list:
            for margin in margin_list:
                start_time = time.time()
                print(f'Start time: {time.strftime("%H:%M:%S", time.gmtime(start_time))}')
                score = CADE_model(ds_name, x_train, y_train, batch_size, similar_samples_ratio, margin, lmbda, val_dataloader)
                end_time = time.time()
                elapsed_time_seconds = end_time - start_time
                elapsed_time_minutes = elapsed_time_seconds / 60
                elapsed_time_hours = elapsed_time_minutes / 60
                print(f'End time: {time.strftime("%H:%M:%S", time.gmtime(end_time))}')
                print(f'Elapsed time for lambda= {lmbda}, margin= {margin}: {elapsed_time_seconds} seconds, {elapsed_time_minutes} minutes, {elapsed_time_hours} hours, loss {score}')
        print('Done: Training For Multiple Centroids')
    model_dir = f'Models/CADE/{ds_name}/'
    best_model_name, _ = find_best_model(x_valid, y_valid, model_dir, Models.Autoencoder)
    print('Done: HyperParameter Search For CADE')
    print(best_model_name)
    model_class = Models.Autoencoder(x_valid.shape[1])
    model_class.load_state_dict(torch.load(f'{best_model_name}'))
    return model_class


def train_classifier(train_loader, model, optimizer, scheduler, epoch):
    model.train()
    error = nn.CrossEntropyLoss()
    train_bar = train_loader
    lr = print_lr(optimizer, epoch)
    total_min_batch_loss = 0
    total_MAE_loss = 0
    correct = 0
    total = 0
    for idx, (img, target) in enumerate(train_bar):
        optimizer.zero_grad()
        labels = target.long().cuda()
        rep= model(img)
        MIN_B_Loss = error(rep, labels)
        total_min_batch_loss += MIN_B_Loss.item()
        total_loss =  MIN_B_Loss
        predicted = torch.max(rep.data, 1)[1]
        total += len(labels)
        correct += (predicted == labels).sum()
        total_loss.backward()
        optimizer.step()
        scheduler.step()
    return model, total_min_batch_loss


def classifier_model(input_dim, train_loader, num_classes, classifier_name, ds_name):
    """
    You can:
    
    (1) use the same hyperparameters we searched for each dataset;
    (2) use the unified model (used as an example in this code);
    (3) re-perform hyperparameter search using the file: SM_HPsearch.py.
    
    """
    best_hp = {
    'CICIDS2017': {'batch': 64,
 'dropout1': 0.6082958021504433,
 'dropout2': 0.7474073738328185,
 'learning_rate': 0.00013911870824876395,
 'neurons1': 128,
 'neurons2': 256,
 'neurons3': 64},
        
    'CICIDS2018': {'batch': 128,
 'dropout1': 0.5127068059873547,
 'dropout2': 0.324020028145305,
 'learning_rate': 0.0004620248072873584,
 'neurons1': 256,
 'neurons2': 64,
 'neurons3': 32},
        
    'unified': {'batch': 512,  'dropout1': 0.1,
 'dropout2': 0.2,
 'learning_rate': 0.0001,
 'neurons1': 128,
 'neurons2': 256,
 'neurons3': 128}
    }
    hp = best_hp['unified']
    lr = hp['learning_rate']
    train_pair_loader = train_loader
    model = Models.SoftmaxClassifier(input_dim,  hp['neurons1'], hp['neurons2'], hp['neurons3'], hp['dropout1'], hp['dropout2'], num_classes).to(device)
    
    val_loss = []
    MAE_losses = []
    train_loss = []
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    total_step = 150 * len(train_pair_loader)
    scheduler = CosineAnnealingLR(optimizer, T_max= total_step, eta_min=lr * 5e-4) 
    assert torch.cuda.is_available()
    cudnn.benchmark = True
    model = nn.DataParallel(model).cuda()
    min_loss = 0
    for epoch in tqdm(range(1, 150+1)):
        model.train()
        train_pair_loader = train_loader
        model, total_min_batch_loss = train_classifier(train_pair_loader,model,optimizer,scheduler,epoch) 
    torch.save(model.module.state_dict(), f'Models/Classifier/{classifier_name}.pth') 
    return model


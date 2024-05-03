import torch
import numpy as np
import pandas as pd
import random
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader, TensorDataset
import DataProcessing as DP
import DetectionUtils as DU
import Metrics
import TrainUtils
import Models

def incremental_windows(data, labels, window_num):
    data_split = np.array_split(data, window_num, axis=0)
    labels_split = np.array_split(labels, window_num, axis=0)
    return data_split, labels_split


def get_drift_unique_labels(y_drift, y_train):
    drift_length = 0
    unique_labels = []
    for i in np.unique(y_drift):
        if i not in np.unique(y_train):
            unique_labels.append(i)
            drift_length = drift_length + len(np.flatnonzero(y_drift == i))
    return np.array(unique_labels), drift_length


def perform_detection(x_data, y_data, y_train, Encoder, centroids, thersholds):
    x = torch.from_numpy(x_data)
    x = torch.tensor(x, dtype=torch.float)
    prd2 = Encoder(x)
    latent_data = prd2.detach().cpu().numpy()
    unique_drift_labels, drift_length = get_drift_unique_labels(y_data, y_train)
    print(unique_drift_labels)
    _, _, _, _, detected_idx, distances = DU.inference(latent_data, centroids, y_data, thersholds)
    
    return detected_idx, distances

def get_farthest_samples(points, labels, k, smpls):
    np.random.seed(0)
    random.seed(0)
    clss_idx = []
    representatives = []
    
    rand_idx = np.random.choice(len(points))
    representatives.append(points[rand_idx])
    clss_idx.append(rand_idx)
    min_distances = np.inf * np.ones(points.shape[0])
    min_distances[rand_idx] = 0

    while len(representatives) < k:
        for rep in representatives:
            distance_vectors = points - rep
            distances = np.linalg.norm(np.abs(distance_vectors), axis=1)
            min_distances = np.minimum(min_distances, distances)
        farth_idx = np.argmax(min_distances)
        representatives.append(points[farth_idx])
        clss_idx.append(farth_idx)
        min_distances[farth_idx] = np.inf

    selected_labels = labels[clss_idx]
    selected_samples = smpls[clss_idx]
    not_selected_x = np.delete(smpls, clss_idx, axis=0)
    not_selected_y = np.delete(labels, clss_idx)
    
    return selected_labels, selected_samples, not_selected_x, not_selected_y

def get_density(thrs, samples, labels, latent):
    k = int(len(samples)/2)
    points_norm = latent / np.linalg.norm(latent, axis=1, keepdims=True)
    dot_prod = np.dot(points_norm, points_norm.T)
    similarty_scores = dot_prod / (np.linalg.norm(latent, axis=1) * np.linalg.norm(latent, axis=1)[:, np.newaxis])
    similarty_sum = np.sum(similarty_scores, axis=1)
    most_similar_idx = np.argpartition(similarty_sum, -k)[-k:]
    most_disimilar_idx = np.argpartition(similarty_sum, k)[:k]
    similar_labels_selected, similar_samples_selected, x_similar_not_selected, y_similar_not_selected = get_farthest_samples(samples[most_similar_idx], labels[most_similar_idx], round(thrs * len(labels[most_similar_idx])), latent[most_similar_idx])
    disimilar_labels_selected, disimilar_samples_selected, x_disimilar_not_selected, y_disimilar_not_selected = get_farthest_samples(samples[most_disimilar_idx], labels[most_disimilar_idx], round(thrs * len(labels[most_disimilar_idx])), latent[most_disimilar_idx])
    selected_samples = np.concatenate((similar_samples_selected,disimilar_samples_selected), axis=0)
    selected_labels = np.concatenate((similar_labels_selected,disimilar_labels_selected))
    notselected_samples = np.concatenate((x_similar_not_selected,x_disimilar_not_selected), axis=0)
    notselected_labels = np.concatenate((y_similar_not_selected, y_disimilar_not_selected))
    return selected_samples, selected_labels, notselected_samples, notselected_labels


def FFT_sample_clusters(clusters, labels, sampling_rate, latent_x):
    unique_labels = []
    labels_all = []
    selected_samples = []
    samples_not_selected = []
    labels_not_selected = []
    
    for i in range(len(clusters)):
        x_data = clusters[i]
        y_data = labels[i]
        samples_x = latent_x[i]
        sampled_samples, sampled_labels, not_selected_x, not_selected_y = get_density(sampling_rate, x_data, y_data, samples_x)
        unique_labels.append(np.unique(sampled_labels))
        labels_all.extend(sampled_labels)
        selected_samples.extend(sampled_samples)
        samples_not_selected.extend(not_selected_x)
        labels_not_selected.extend(not_selected_y)

    return unique_labels, labels_all, selected_samples, labels_not_selected, samples_not_selected

def relabel_data(train_y, Y_test):
    unique_train = np.unique(train_y)
    unique_test = np.unique(Y_test)
    non_shared_labels = np.setdiff1d(unique_test, unique_train)
    for label in non_shared_labels:
        Y_test[Y_test == label] = label + 100
    shared_labels = np.intersect1d(unique_train, unique_test)
    mapping = {label: idx for idx, label in enumerate(shared_labels)}

    for label, new_label in mapping.items():
        train_y[train_y == label] = new_label
        Y_test[Y_test == label] = new_label

    print("train_y", np.unique(train_y))
    print("Y_test", np.unique(Y_test))
    return train_y, Y_test


def clf_metrics(notselected_y,prds):
    accuracy = accuracy_score(notselected_y, prds) * 100
    print(f'Classifier Accuracy {accuracy}')
    b_accuracy = balanced_accuracy_score(notselected_y, prds) * 100
    print(f'Classifier Balanced Accuracy {b_accuracy}')
    f1 = f1_score(notselected_y, prds, average='macro', zero_division=0) * 100
    print(f'Classifier macro F1 {f1}')
    precision = precision_score(notselected_y, prds,average='macro', zero_division=0) * 100
    print(f'Classifier macro Precision {precision}')


def random_split(x_data, y_data, pesudo_labels, selection_rate):
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    pesudo_labels = np.array(pesudo_labels)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    unique_pesudo_labels = np.unique(pesudo_labels)
    print(unique_pesudo_labels)
    for label in unique_pesudo_labels:
        label_idx = np.where(pesudo_labels == label)[0]
        true_labels = y_data[label_idx]
        label_samples = x_data[label_idx]
        label_pesudo_labels = pesudo_labels[label_idx]
        k = int(selection_rate * len(label_samples))
        selected_idx = np.random.choice(np.arange(len(label_samples)), size=k)
        selected_samples = label_samples[selected_idx]
        selected_labels = label_pesudo_labels[selected_idx]
        not_selected_x = np.delete(label_samples, selected_idx, axis=0)
        not_selected_y = np.delete(true_labels, selected_idx)
        x_train.append(selected_samples)
        y_train.append(selected_labels)
        x_test.append(not_selected_x)
        y_test.append(not_selected_y)
    
    return np.vstack(x_train), np.vstack(x_test), np.hstack(y_train), np.hstack(y_test)


def DNN_Pesudo_Labels(x_test, model):
    model.eval()
    rep = model(x_test)
    predicted = torch.max(rep.data, 1)[1]
    y_pred = predicted.detach().cpu().numpy()
    return y_pred

def Selection_fn(All_set, All_labels, wind_len, SR, Encoder, init_num_classes):
    All_set_latent = Encoder(torch.tensor(All_set, dtype=torch.float)).detach().cpu().numpy()
    data_divided, labels_divide = incremental_windows(All_set_latent, All_labels, int(len(All_set) / wind_len))
    data_divided1, labels_divide1 = incremental_windows(All_set, All_labels, int(len(All_set) / wind_len))
    uniq_labels, selected_labels, selected_samples, labels_not_selected, samples_not_selected = FFT_sample_clusters(data_divided, labels_divide, SR, data_divided1)
    selected_labels = np.array(selected_labels)
    labels_not_selected = np.array(labels_not_selected)
    mask = selected_labels < init_num_classes
    selected_labels[mask] = 0 
    mask2 = labels_not_selected < init_num_classes
    labels_not_selected[mask2] = 0
    y_lab = np.unique(selected_labels)[np.unique(selected_labels) > (init_num_classes-1)]
    y_all = np.unique(All_labels)[np.unique(All_labels) > (init_num_classes-1)]
    div = len(np.unique(y_lab)) / len(np.unique(y_all))* 100
    print(f'Diversity {div}')
    return selected_labels, selected_samples, labels_not_selected, samples_not_selected

def get_pesudo_labels(selected_samples, selected_labels, notselected_x, notselected_y):
    print('Random Forest As an Oracle (i.e., using it to label the non-labeled set)')
    RF = RandomForestClassifier(random_state=0)
    RF = RF.fit(selected_samples, selected_labels)
    RFprds = RF.predict(notselected_x)
    print('Random Forest Results on Non-Selected Data (The pseudo labels)..')
    Metrics.clf_metrics(notselected_y, RFprds)
    print('________________________________________')
    return RFprds
    
def DNN_Pesudo_Labels(x_test, model):
    model.eval()
    rep = model(x_test)
    predicted = torch.max(rep.data, 1)[1]
    y_pred = predicted.detach().cpu().numpy()
    return y_pred

def perform_adaptation(x_train, y_train, x_test, y_test, Encoder, SR, acceptance_error, train_new=True, DS="CICIDS17", wind_len=1000):

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
    init_num_classes = len(np.unique(y_train))
    init_train_loader =  DP.Rasd_loader(x_train, y_train, batch_size=512)
    if train_new == True:
        init_classifier = TrainUtils.classifier_model(x_train.shape[1], init_train_loader, init_num_classes, f'{DS}-Init', DS)
    else: 
        init_classifier = Models.SoftmaxClassifier(x_train.shape[1],  hp['neurons1'], hp['neurons2'], hp['neurons3'], hp['dropout1'], hp['dropout2'], init_num_classes)
        init_classifier.load_state_dict(torch.load(f'Models/Classifier/{DS}-Init.pth'))
    
    x_train_tensor = torch.from_numpy(x_train).float()
    projections = Encoder(x_train_tensor).detach().cpu().numpy()
    latent, labels = pd.DataFrame(projections), pd.DataFrame(y_train) 
    latent["Label"] = labels
    centriods, averages = DU.samples_distance_to_centroids(latent, init_num_classes)
    thersholds, error_rates = DU.adjust_thresholds(latent, centriods, averages, acceptance_error)
    detected_idx, distances = perform_detection(x_test, y_test, y_train, Encoder, centriods, thersholds)
    All_set = x_test[detected_idx]
    All_labels = y_test[detected_idx]
    selected_labels, selected_samples, labels_not_selected, samples_not_selected = Selection_fn(All_set, All_labels, wind_len, SR, Encoder, init_num_classes)
    pesudo_labels = get_pesudo_labels(selected_samples, selected_labels, samples_not_selected, labels_not_selected)
    X_train, X_test, Y_train, Y_test = random_split(samples_not_selected, labels_not_selected, pesudo_labels, 0.7) 
    Known_set_idx = np.where(Y_train == 0)
    Known_set = torch.from_numpy(X_train[Known_set_idx]).float()
    Known_Pesudo_Labels = DNN_Pesudo_Labels(Known_set, init_classifier)
    Y_train[Known_set_idx] = Known_Pesudo_Labels
    train_x = np.concatenate((selected_samples, X_train, x_train), axis=0)
    train_y = np.concatenate((selected_labels,Y_train, y_train))
    X_test = np.concatenate((X_test, x_test), axis=0)
    Y_test = np.concatenate((Y_test, y_test))
    train_y, Y_test = relabel_data(train_y, Y_test)
    X_train = torch.from_numpy(train_x).float()
    Y_train = torch.from_numpy(np.array(train_y))
    evaluates_multiple_classifiers(X_train, Y_train, X_test, Y_test, DS)


def evaluates_multiple_classifiers(X_train, Y_train, X_test, Y_test, DS, mode="Train"):
    print('DNN')
    ntrain_dataset = DataLoader(dataset=TensorDataset(X_train, Y_train), batch_size=512, shuffle=True, num_workers=2)
    Snd_classifier = TrainUtils.classifier_model(X_train.shape[1], ntrain_dataset, len(np.unique(Y_train)), DS, DS)
    print('The DNN Classifier Results on Test Data (After Adaptation) ..')
    accuracy, b_accuracy, f1, precision, wf1, wpre = Metrics.evaluate_classifier(Y_test, torch.from_numpy(X_test).float(), Snd_classifier)
    print('________________________________________')
    return 

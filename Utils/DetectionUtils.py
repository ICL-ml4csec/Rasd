import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import Metrics

def samples_distance_to_centroids(projections, num_classes):
    centriods = [] 
    averages = []
    for i in range(num_classes):
        samples = projections.query(f'Label == {i}')
        x_samples = samples.loc[:, samples.columns != "Label"]
        x_samples = np.array(x_samples)
        centroid = np.mean(x_samples, axis=0)
        centriods.append(centroid)
        list_of_distances = []
        for j in x_samples:
            dis = (np.linalg.norm(np.abs(j - centroid)))
            list_of_distances.append(dis)
        avg_dis = np.mean(list_of_distances)
        averages.append(avg_dis)
    centriods = np.array(centriods)
    averages = np.array(averages)
    print(' Done: Centroids Identification')
    return centriods, averages

def adjust_thresholds(projections, centroids, averages, accepted_percent):
    thresholds = []
    error_rates = []
    for i, (centroid, avg) in enumerate(zip(centroids, averages)):
        distances = np.linalg.norm(projections[projections['Label'] == i].drop('Label', axis=1).to_numpy() - centroid, axis=1)
        threshold = avg
        while np.mean(distances > threshold) > accepted_percent:
            threshold += avg * 0.01
        thresholds.append(threshold)
        error_rate = np.mean(distances > threshold) * 100
        print(f'class {i} -- threshold {threshold} -- detected samples % {error_rate}')
    return np.array(thresholds), np.array(error_rates)


def inference(data, centroids, labels, thersholds):
    detected_drifts = 0 
    not_detected = 0
    detected_classes = []
    detected_samples = []
    drift_idx = []
    distances = []
    for i in tqdm(range(len(data))):
        dis_k = [np.linalg.norm(np.abs(data[i] - centroids[j])) for j in range(len(centroids))]
        true = 0
        for k in range(len(dis_k)):
            if dis_k[k] > thersholds[k]:
                true = true + 1
        if true > (len(centroids)-1):
            detected_drifts = detected_drifts+1
            detected_classes.append(labels[i])
            detected_samples.append(data[i])
            drift_idx.append(i)
            distances.append(np.min(dis_k))
        else:
            not_detected = not_detected+1
            detected_classes.append(0)
    return detected_drifts, not_detected, detected_classes, np.array(detected_samples), np.array(drift_idx), distances


def Detection_and_Results(x_non_drift, y_non_drift, x_drift, y_drift, x_test, y_test, x_train, y_train, model, acceptance_error):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    def to_tensor(data):
        if not torch.is_tensor(data):
            data = torch.tensor(data, dtype=torch.float).to(device)
        else:
            data = data.to(device)
        return data
    model.eval()
    x_train = to_tensor(x_train)
    projections = model(x_train).detach().cpu().numpy()
    latent, labels = pd.DataFrame(projections), pd.DataFrame(y_train)
    latent["Label"] = labels
    centroids, averages = samples_distance_to_centroids(latent, len(np.unique(y_train)))
    thresholds, error_rates = adjust_thresholds(latent, centroids, averages, acceptance_error)
    print(' Done: Threshold Search')
    x_non_drift = to_tensor(x_non_drift)
    prd2 = model(x_non_drift)
    z_non_drift = prd2.detach().cpu().numpy()
    ntest_detected, ntest_no_detected, _, _, _, _ = inference(z_non_drift, centroids, np.array(y_non_drift), thresholds)
    print(f'Done: x_non_drift Prediction')

    
    x_drift = to_tensor(x_drift)
    prd2 = model(x_drift)
    z_drift = prd2.detach().cpu().numpy()
    drift_detected, drift_not_detected, detected_classes, _, _, _ = inference(z_drift, centroids, list(y_drift), thresholds)
    print(F'Done: x_drift Prediction')
    print(drift_detected)
    print(drift_not_detected) 
    x_test = to_tensor(x_test)
    prd2 = model(x_test)
    z_test = prd2.detach().cpu().numpy()
    test_detected, test_no_detected, _, _, _, _ = inference(z_test, centroids, y_test, thresholds)
    print(f'Done: x_test Prediction')
    TP = drift_detected
    TN = test_no_detected + ntest_no_detected
    FP = test_detected + ntest_detected
    FN = drift_not_detected
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f'Shift Detection Metrics')
    Metrics.evaluates_drift(y_drift, TP, TN, FP, FN, detected_classes)
    print(f' Latent Representations Quality')
    #Metrics.latent_quality(x_test, y_test, model)
    return TP, TN, FP, FN


from scipy.spatial import distance
from scipy.stats import median_abs_deviation
import Metrics
import torch
import numpy as np
import random


seed = 0
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(0)

device = "cuda" if torch.cuda.is_available() else "cpu"



def to_tensor(data):
    if not torch.is_tensor(data):
        data = torch.tensor(data, dtype=torch.float).to(device)
    else:
        data = data.to(device)
    return data

def predict(x_test, model):
    model.eval()
    x_test = to_tensor(x_test)
    x_test = x_test.cuda()
    if isinstance(model, torch.nn.DataParallel):
        z_test = model.module.encoder(x_test)
    else:
        z_test = model.encoder(x_test)
    return z_test.detach().cpu().numpy()


def get_latent_data_for_each_family(z_train, y_train):
    N = len(np.unique(y_train))
    N_family = [len(np.where(y_train == family)[0]) for family in range(N)]
    z_family = []
    for family in range(N):
        z_tmp = z_train[np.where(y_train == family)[0]]
        z_family.append(z_tmp)
    z_len = [len(z_family[i]) for i in range(N)]
    print(f'z_family length: {z_len}')

    return N, N_family, z_family

def get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family):
    dis_family = []  
    for i in range(N): 
        dis = [np.linalg.norm(z_family[i][j] - centroids[i]) for j in range(N_family[i])]
        dis_family.append(dis)
    dis_len = [len(dis_family[i]) for i in range(N)]
    print(f'dis_family length: {dis_len}')

    return dis_family

def get_MAD_for_each_family(dis_family, N, N_family):
    mad_family = []
    median_list = []
    for i in range(N):
        median = np.median(dis_family[i])
        median_list.append(median)
        print(f'family {i} median: {median}')
        diff_list = [np.abs(dis_family[i][j] - median) for j in range(N_family[i])]
        mad = 1.4826 * np.median(diff_list)  # 1.4826: assuming the underlying distribution is Gaussian
        mad_family.append(mad)
    print(f'mad_family: {mad_family}')

    return mad_family, median_list


def drift_detection(X_test, y_test, z_test, thres, centroids, dis_family, mad_family):
    centroids_array = np.array(centroids)  
    dis_matrix = distance.cdist(z_test, centroids_array, 'euclidean')
    
    dis_k_minus_dis_family = dis_matrix - np.array(dis_family)
    anomaly_k = np.abs(dis_k_minus_dis_family) / np.array(mad_family)

    closest_family = np.argmin(dis_matrix, axis=1)
    min_dis = np.min(dis_matrix, axis=1)
    min_anomaly_score = np.min(anomaly_k, axis=1)

    detected_mask = min_anomaly_score > thres
    detected_drift = np.sum(detected_mask)
    not_detected = len(X_test) - detected_drift

    detected_classes = np.zeros_like(y_test)  
    detected_classes[detected_mask] = y_test[detected_mask]  
    return detected_drift, not_detected, detected_classes


def threshold_search(x_train, y_train, latent_set, error_rate, centroids, dis_family, mad_family):
    lower_bound = 1.0  
    upper_bound = 50.0  
    target_error_rate = error_rate  
    detected, not_detected, _ = drift_detection(x_train, y_train, latent_set, 3.5, centroids, dis_family, mad_family)
    error_rate = detected/(detected+not_detected)
    print(f'Original threshold (3.5) proposed by CADE emperically -- Error rate {error_rate * 100}')
    while upper_bound - lower_bound > 0.001:  
        mid_point = (upper_bound + lower_bound) / 2.0
        detected, not_detected, _ = drift_detection(x_train, y_train, latent_set, mid_point, centroids, dis_family, mad_family)
        error_rate = detected / (detected + not_detected)
        if error_rate > target_error_rate:
            lower_bound = mid_point
        else:
            upper_bound = mid_point
    print(f'Final threshold is {mid_point} for an error rate of approximately {error_rate*100}%')
    return mid_point


def eval_CADE(x_non_drift, y_non_drift, x_drift, y_drift, x_test, y_test, x_train, y_train, model, acceptance_error):
    model.to(device)
    z_train = predict(x_train, model)
    test_latent = predict(x_test, model)
    nodrift_latent = predict(x_non_drift, model)
    drift_latent = predict(x_drift, model)
    N, N_family, z_family = get_latent_data_for_each_family(z_train, y_train)
    centroids = [np.mean(z_family[i], axis=0) for i in range(N)]
    print(f'centroids: {centroids}')
    dis_family = get_latent_distance_between_sample_and_centroid(z_family, centroids, N, N_family)
    mad_family, dis_family = get_MAD_for_each_family(dis_family, N, N_family)
    X_latent = predict(x_train, model)
    thres = threshold_search(x_train, y_train, X_latent, acceptance_error, centroids, dis_family, mad_family)
    drift_detected, drift_not_detected, detected_classes = drift_detection(x_drift, np.array(y_drift), drift_latent, thres, centroids, dis_family, mad_family)
    test_detected, tested_not_detected, _ = drift_detection(x_test, y_test, test_latent, thres, centroids, dis_family, mad_family)
    nodrift_detected, nodrift_not_detected, _ = drift_detection(x_non_drift,y_non_drift, nodrift_latent, thres, centroids, dis_family, mad_family)
    TP = drift_detected
    TN = tested_not_detected + nodrift_not_detected
    FP = test_detected + nodrift_detected
    FN = drift_not_detected
    print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
    print(f'Shift Detection Metrics')
    Metrics.evaluates_drift(y_drift, TP, TN, FP, FN, detected_classes)

    print(f'Latent Representations Quality')
    #Metrics.latent_quality(x_test, y_test, model)
    return TP, TN, FP, FN

from scipy.spatial.distance import cdist
import numpy as np
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, precision_score, classification_report
import torch 

def dunn_index(X, labels):
    clusters = np.unique(labels)
    max_intra_cluster_distance = -np.inf
    min_inter_cluster_distance = np.inf
    
    for cluster in clusters:
        cluster_points = X[labels == cluster]
        intra_distances = cdist(cluster_points, cluster_points, 'euclidean')
        np.fill_diagonal(intra_distances, -np.inf)
        max_distance = np.max(intra_distances)
        if max_distance > max_intra_cluster_distance:
            max_intra_cluster_distance = max_distance
        for other_cluster in clusters:
            if cluster != other_cluster:
                other_cluster_points = X[labels == other_cluster]
                inter_distances = cdist(cluster_points, other_cluster_points, 'euclidean')
                min_distance = np.min(inter_distances)
                if min_distance < min_inter_cluster_distance:
                    min_inter_cluster_distance = min_distance

    if max_intra_cluster_distance > 0:  # To prevent division by zero
        return min_inter_cluster_distance / max_intra_cluster_distance
    else:
        return 0

def evaluate_clustering(x, y):
    kmeans = KMeans(n_clusters=len(np.unique(y)), init='k-means++', random_state=0).fit(x)
    y_pred = kmeans.labels_
    dunn_result = dunn_index(x, y_pred)
    print("Dunn Index:", dunn_result)
    return dunn_result


def macro_recall(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    num_labels = np.unique(y_true)
    recalls = []
    for label in num_labels:
        true_positive = np.sum((y_pred == label) & (y_true == label))  
        actual_positive = np.sum(y_true == label)  # 
        if actual_positive == 0:  # to avoid division by zero
            recalls.append(0)
        else:
            recall = true_positive / actual_positive
            recalls.append(recall)
    return np.mean(recalls)


def evaluates_drift(y_drift, TP, TN, FP, FN, detected_classes):
    Acc = (TP + TN) / (TP + TN + FP + FN)
    print("Drift Detection Accuracy: ", Acc)
    
    Recall = TP / (TP + FN) if TP + FN > 0 else 0
    print('Micro Recall:', Recall)
    
    diversity = (len(np.unique(detected_classes)) - 1) / len(np.unique(y_drift)) if len(np.unique(y_drift)) > 0 else 0
    print("Diversity Score: ", diversity)
    
    Macro_recall = macro_recall(y_drift, detected_classes)
    print('Macro Recall:', Macro_recall)
    
    Precision = TP / (TP + FP) if TP + FP > 0 else 0
    print("Precision: ", Precision)
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall) if (Precision + Recall) > 0 else 0
    print("F1 Score: ", F1)
    
    TNR = TN / (FP + TN) if FP + TN > 0 else 0
    FPR = FP / (FP + TN) if FP + TN > 0 else 0
    print("False Positive Rate (FPR): ", FPR * 100)
    print("True Negative Rate (TNR): ", TNR * 100)
    
    return Recall * 100, Macro_recall * 100, Precision * 100, F1 * 100, Acc * 100, diversity * 100


def clustering_metics(x, y, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(x)
    cluster_labels = kmeans.labels_
    print('clustering done')

    def calculate_metric(metric, x, y, cluster_labels):
        if metric == 'silhouette':
            result = metrics.silhouette_score(x, cluster_labels)
        elif metric == 'nmi':
            result = metrics.normalized_mutual_info_score(y, cluster_labels)
        elif metric == 'accuracy':
            contingency_matrix = metrics.cluster.contingency_matrix(y, cluster_labels)
            row_ind, col_ind = linear_sum_assignment(-contingency_matrix)
            new_cluster_labels = np.empty_like(cluster_labels)
            for i in range(n_clusters):
                new_cluster_labels[cluster_labels == col_ind[i]] = row_ind[i]
            result = accuracy_score(y, new_cluster_labels)
        else:
            result = None
        print(f'{metric} done')
        return result
    metrics_list = ['silhouette', 'nmi', 'accuracy']
    results = []
    for metric in metrics_list:
        results.append(calculate_metric(metric, x, y, cluster_labels))

    return tuple(results)


def latent_quality(x_test, y_test, model):
    x_test = torch.tensor(x_test, dtype=torch.float32)
    y_tes = torch.tensor(y_test, dtype=torch.long)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_lat = model(x_test.to(device)).detach().cpu().numpy()
    silhouette, nmi, accuracy = clustering_metics(test_lat, y_test,
                            n_clusters=len(np.unique(y_test)))
    print('Silhouette Score:', silhouette)
    print('nmi Score:', nmi)
    print('Accuracy:', accuracy)
    return 


def evaluate_classifier(y_test, x_test, model):
    model.eval()
    rep = model(x_test)
    predicted = torch.max(rep.data, 1)[1]
    y_pred = predicted.detach().cpu().numpy()
    accuracy = accuracy_score(y_test, y_pred) * 100
    print(f'Classifier Accuracy {accuracy}')
    b_accuracy = balanced_accuracy_score(y_test, y_pred) * 100
    print(f'Classifier Balanced Accuracy {b_accuracy}')
    f1 = f1_score(y_test, y_pred, average='macro') * 100
    print(f'Classifier F1 {f1}')
    wf1 = f1_score(y_test, y_pred, average='weighted') * 100
    precision = precision_score(y_test, y_pred, average='macro') * 100
    wpre = precision_score(y_test, y_pred, average='weighted') * 100
    print(f'Classifier Precision {precision}')
    print(classification_report(y_test, y_pred))
    return accuracy, b_accuracy, f1, precision, wf1, wpre

def clf_metrics(notselected_y, prds):
    accuracy = accuracy_score(notselected_y, prds) * 100
    print(f'Classifier Accuracy {accuracy}')
    b_accuracy = balanced_accuracy_score(notselected_y, prds) * 100
    print(f'Classifier Balanced Accuracy {b_accuracy}')
    f1 = f1_score(notselected_y, prds, average='macro', zero_division=0) * 100
    print(f'Classifier macro F1 {f1}')
    precision = precision_score(notselected_y, prds, average='macro', zero_division=0) * 100
    print(f'Classifier macro Precision {precision}')

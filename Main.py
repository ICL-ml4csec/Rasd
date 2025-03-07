import numpy as np
import pandas as pd
import sys
import argparse

sys.path.append('Utils/')
import CentroidDefinition as CD
import Models
import DataProcessing as DP
import CLlosses
import TrainUtils
import DetectionUtils as DU
import AdaptationUtils
import CADEutils


parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str, default="CICIDS2017", choices=["CICIDS2017", "CICIDS2018"])

parser.add_argument('--acceptance_err', type=float, default=0.07, choices = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1], help="What is the threshold for shift detection (i.e., k-percentile)?")

parser.add_argument('--train_mode', type=str, default="pre-train", choices = ["pre-train", "train-new"],
                    
                    help="Would you train from scratch (train-new)? or use already trained models (pre-train)?")

parser.add_argument('--Mode', type=str, default="Both", choices = ["Detection", "Both"],
                    help="Would you detect shift only (Detection)?, or also adapt to it (Both)?.")

parser.add_argument('--Detection_Method', type=str, default="Rasd", choices = ["Rasd", "LSL", "CADE"],
                    help="The method that will be used for shift detection.")
                    

parser.add_argument('--selection_rate', type=float, default=0.05, choices = [0.01, 0.02, 0.03, 0.04, 0.05], help="How many samples would you select from the detected shift samples for human labeling?")


parser.add_argument('--selection_batch_size', type=int, default=3000, choices = [3000, 2000, 1000], help="The selection method batch size (refer to section 3.2 in the paper)")


args = parser.parse_args()



x_train, x_valid, x_test, x_drift, x_non_drift, y_train, y_valid, y_test, y_drift, y_non_drift =  DP.data_processing(args.dataset_name)

def perform_detection(args, x_train, x_valid, y_train, y_valid, x_non_drift, y_non_drift, x_drift, y_drift):
    num_classes = len(np.unique(y_train))
    print(f'Num Classes {num_classes}')
    num_features = (x_train.shape[1])
    latent_size = int(num_features*0.1)
    print(f' Output size would be {latent_size}')
    if args.Detection_Method == "Rasd":
        Best_Model = TrainUtils.Rasd_HP(x_train, y_train, x_valid, y_valid, latent_size, num_classes, args.dataset_name, args.train_mode)
        DU.Detection_and_Results(x_non_drift, y_non_drift, x_drift, y_drift, x_test, y_test, x_train, y_train, Best_Model, args.acceptance_err)
    elif args.Detection_Method == "LSL":
        Best_Model = TrainUtils.LSL_HP(x_train, y_train, x_valid, y_valid, latent_size, num_classes, args.dataset_name, args.Detection_Method, args.train_mode)
        DU.Detection_and_Results(x_non_drift, y_non_drift, x_drift, y_drift, x_test, y_test, x_train, y_train, Best_Model, args.acceptance_err)
    elif args.Detection_Method == "CADE":
        Best_Model = TrainUtils.CADE_HP(x_train, y_train, x_valid, y_valid, args.dataset_name, args.train_mode)
        CADEutils.eval_CADE(x_non_drift, y_non_drift, x_drift, y_drift, x_test, y_test, x_train, y_train, Best_Model, args.acceptance_err)
    return Best_Model

Best_Model = perform_detection(args, x_train, x_valid, y_train, y_valid, x_non_drift, y_non_drift, x_drift, y_drift)

if args.Mode == "Both" and args.Detection_Method == "Rasd":
    X_test = np.concatenate((x_drift, x_non_drift, x_test), axis=0)
    Y_test = np.concatenate((y_drift, y_non_drift, y_test), axis=0)
    AdaptationUtils.perform_adaptation(x_train, y_train, X_test, Y_test, Best_Model.to('cpu'), args.selection_rate, args.acceptance_err, DS=args.dataset_name, wind_len=args.selection_batch_size)

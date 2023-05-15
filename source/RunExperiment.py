# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 18:52:22 2023

@author: Song Keyu
"""

import numpy as np
import argparse
import pickle
import os
from os.path import join
import settings



def get_params(): 
    parser = argparse.ArgumentParser(description='SelfATPU')
    parser.add_argument('--dataset_name', type=str, default='DAMADICS', choices=['TEP','DAMADICS'])
    parser.add_argument('--method',  type=str,   default = "SatPU", choices=['SatPU','DeepDCR','Baseline','DistPU'])
    parser.add_argument('--F_ratio', type=float, default=0.2, help='ratio of fault samples in training and test set (only for TEP)')
    parser.add_argument('--P_ratio', type=float, default=0.4, help='ratio of labeled positive samples in training set')
    parser.add_argument('--length',  type=int,   default=12,  help = 'length of each sample')
    parser.add_argument('--step',    type=int,   default=1,   help = 'Sampling step')
    parser.add_argument('--caseid',  type=str,   default="",  help = 'identifier in folder name for each test run')
    
    parser.add_argument('--fault_number',  type=int,   default=1,   help = 'Fault id in 1,2, ... 20 for TEP (IDV)')
    parser.add_argument('--actuator',      type=str,   default="Actuator3",choices=["Actuator1","Actuator2","Actuator3","All"],help = 'actuator typr for DAMADICS')
    parser.add_argument('--sat_momentum',  type=float, default=0.8, help = 'Momentum in (0,1] for SatPU')
    parser.add_argument('--ini_epochs',    type=int,   default=20,  help = 'Total epochs in initilization stage of SatPU')
    parser.add_argument('--sat_epochs',    type=int,   default=60,  help = 'Total epochs in self-adpative training stage of SatPU')
    parser.add_argument('--filter_window', type=int,   default=5,   help = 'Window length of temporal median filter of SatPU')
    parser.add_argument('--PSEUDO_LABELING'          , type=int, default=1,choices=[0,1],help="Use PSEUDO_LABELING step in SatPU")
    parser.add_argument('--NONELINEAR_REWEIGHTING'   , type=int, default=1,choices=[0,1],help="Use NONELINEAR_REWEIGHTING step in SatPU")
    parser.add_argument('--AMBIGUOUS_INITIALIZATION' , type=int, default=1,choices=[0,1],help="Use AMBIGUOUS_INITIALIZATION step in SatPU")
    parser.add_argument('--TEMPORAL_FILTER'          , type=int, default=1,choices=[0,1],help="Use TEMPORAL_FILTER step in SatPU")
    
    parser.add_argument('--load_saved',    type=bool,  default=False,  help = 'Load saved training set of TEP dataset (faster)')
    
    args = parser.parse_args()
    print(args)

    return args


if __name__ == '__main__':
    try:
        args = get_params()

        ### Load Dataset
        if args.dataset_name == "DAMADICS":
            from DAMADICS_Dataset import TrainingSet1109,TestSet1117
            X,Y,scaler = TrainingSet1109(1-args.P_ratio,args.length,args.step,"Actuator3")
            Y = np.array([Y,1-Y]).T
            Xt,Yt = TestSet1117(scaler,args.length,args.step,"Actuator3")
        elif args.dataset_name == "TEP":
            from TEP_Dataset import LoadDataSet,ModifyTrainingData,TEP_Windowed_DataSet
            ## TODO
            if args.load_saved:
                print ("Loading existing TEP training package")
                x_train,y_train,x_test,y_test = LoadDataSet(join(settings.DATA_DIR,"IDV(%d).pkl"%(args.fault_number)))
            else:
                print ("Creating TEP training set IDV(%d)"%(args.fault_number))
                x_train,y_train,x_test,y_test=TEP_Windowed_DataSet(
                    fault_number    = args.fault_number,
                    input_length    = args.length,
                    step            = args.step,
                    subset          = 1.0,
                    ambigous        = 0.0)

            
            X,Y = ModifyTrainingData(x_train,y_train,4000,args.F_ratio,args.P_ratio)
            
            t_norm = 10000
            Xt = np.concatenate([x_test[:t_norm,:,:], x_test[-int(t_norm*args.F_ratio):,:,:]])
            Yt = np.concatenate([y_test[:t_norm,:]  , y_test[-int(t_norm*args.F_ratio):,:]])
            Yt = Yt[:,0]
         
        ### Initialize Model
        feature_extractor = None
        if args.dataset_name == "TEP":
            if args.method == "DeepDCR":
                from LSTM_regression import LSTM_classifier
                feature_extractor = LSTM_classifier(18,args.length,X.shape[2])
            else:
                from LSTM_regression import LSTM_regression
                feature_extractor = LSTM_regression(18,args.length,X.shape[2])
        elif args.dataset_name == "DAMADICS":
            if args.method == "DeepDCR":
                from MLP_regression import MLP_classifier
                feature_extractor = MLP_classifier(args.length*X.shape[2],args.length,X.shape[2])
            else:
                from MLP_regression import MLP_regression
                feature_extractor = MLP_regression(args.length*X.shape[2],args.length,X.shape[2])
        
        if args.method == "SatPU":
            from SatPU import SatPU
            model = SatPU(regression                    = feature_extractor,
                          momentum                      = args.sat_momentum,
                          filter_window                 = args.filter_window,
                          IniEpochs                     = args.ini_epochs,
                          SatEpochs                     = args.sat_epochs,
                          USE_PSEUDO_LABELING           = args.PSEUDO_LABELING,
                          USE_NONELINEAR_REWEIGHTING    = args.NONELINEAR_REWEIGHTING,
                          USE_AMBIGUOUS_INITIALIZATION  = args.AMBIGUOUS_INITIALIZATION,
                          USE_TEMPORAL_FILTER           = args.TEMPORAL_FILTER)
            print (model.AblationCode())
        elif args.method == "Baseline":
            from BaseLine import Baseline
            model = Baseline(feature_extractor)
        elif args.method == "DeepDCR":
            from DeepDCR import DeepDCR
            model = DeepDCR(feature_extractor)
            
        ### Train
        if args.method == "SatPU" or args.method == "Baseline":
            Y_TRAIN = Y
        elif args.method == "DeepDCR":
            Y_TRAIN = (Y[:,0]==True)
        model.fit(X,Y_TRAIN,False,Xt)
    
        ### Test
        y_pred=model.predict(Xt)
        
        
        ### Experiment Identity String
        method_name = args.method
        if args.method=="SatPU":
            method_name += "_" + model.AblationCode()
        if args.dataset_name == "TEP":
            DatasetCode = "TEP-%d"%(args.fault_number)
        elif args.dataset_name == "DAMADICS":
            DatasetCode = "DAM-09-17#%s"%(args.actuator[-1])
        title = "%s(%s)[F%d%%P%d%%]%s"%(
            method_name,DatasetCode,
            args.F_ratio*100,args.P_ratio*100,
            args.caseid)
        
        
        ### save model and result
        savepath = join(settings.MODEL_DIR,"%s"%(title))
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        try:
            feature_extractor.save(savepath)
        except:
            print ('Save model Failure')
        with open(os.path.join(savepath,"validation.pkl"),"wb") as output:
            pickle.dump(model.result_valid, output)
        
        import matplotlib.pyplot as plt
        import matplotlib
        try:
            matplotlib.use('Qt5Agg')
        except:
            print("using Qt5Agg Failed")
        plt.plot(y_pred[:,0])
        # plt.plot(Xt[:,0,-5])
        plt.plot(Yt)
        Accuracy  = np.sum((y_pred[:,1]>0.5) ^ (Yt==1)) / len(Yt)
        Precision = np.sum((y_pred[:,1]>0.5) & (Yt==0)) / np.sum(y_pred[:,1]>0.5)
        Recall    = np.sum((y_pred[:,1]>0.5) & (Yt==0)) / np.sum(Yt==0)
        F1        = 2.0* Precision * Recall / (Precision + Recall)
        print("%.2f"%(Accuracy*100))
        print("%.2f"%(Precision*100))
        print("%.2f"%(Recall*100))
        print("%.2f"%(F1*100))
    except Exception as exception:
        print(exception)
        raise
    from numba import cuda
    cuda.select_device(0)
    cuda.close()

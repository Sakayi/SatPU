# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 17:19:28 2023

@author: Song Keyu
"""


import numpy as np
import sys
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

sys.path.append("../../code-download/DeepDCR/lib") ## TODO add lib
from gcforest.gcforest import GCForest

import settings
from os.path import join,exists
from os import makedirs


class DeepDCR:
    def __init__(self,
                 classifier,
                 IniEpochs = 40,
                 NxtEpochs = 40,
                 config = {"random_state": 1,
                           "max_layers":   100,
                           "early_stopping_rounds": 3,
                           "n_classes" :   2,
                           "cascade"   :   {},}):
        self.config         = config
        self.classifier     = classifier
        self.IniEpochs      = IniEpochs
        self.NxtEpochs      = NxtEpochs
        self.result_valid   = None
        #self.gc2 = GCForest(config)
    
    def validate(self,x_valid,epoch_id):
        if self.result_valid is None:
            self.result_valid = np.zeros((self.IniEpochs+self.NxtEpochs,x_valid.shape[0]))
        self.result_valid[epoch_id] = self.predict(x_valid)[:,1]
        
    
    
    def fit(self,train_x,train_y,show=False,x_validate = None):
        x_p = train_x[train_y==True]
        y_p = train_y[train_y==True]
        x_u = train_x[train_y!=True]
        # print (x_u.shape)
        X_n = x_u[-1]
        y_u = train_y[train_y!=True]
        
        x_u_u, x_u_m, y_u_u, y_u_m = train_test_split(x_u, y_u, test_size=0.2)
        x = np.concatenate((x_p, x_u_m), axis=0)
        y = np.concatenate((y_p, y_u_m), axis=0)

        # scaler = StandardScaler().fit(x)
        # x_train_transformed = scaler.transform(x)
        # x_u_train_transformed = scaler.transform(x_u_u)
        x_train_transformed = x
        x_u_train_transformed = x_u_u
        
        ### Find Reliable Negative
        config = self.config
        model  = self.classifier
        
        ## model.fit_transform(x_train_transformed, y,x_validate)
        
        
        if x_validate is None:
            model.fit_transform(x_train_transformed,y,
                      epochs=self.IniEpochs,sample_weight=None)
        else:
            for i in range(self.IniEpochs):
                model.fit_transform(x_train_transformed,y,
                          epochs=1,sample_weight=None)
                self.validate(x_validate,i)
        
        scores = model.predict_proba(x_u_train_transformed)[:, 1]
        orderScores = np.argsort(scores)
        orderList = [str(item) for item in orderScores]
        orderStr = ','.join(orderList)
        top = int(y_u.shape[0] * 0.1)
        topNIndex = orderScores[:top]
        # print (topNIndex)
        # t = 0
        # while t < top:
        #     index = topNIndex[t]
        #     x_n = x_u_u[index]
        #     X_n = np.vstack((X_n, x_n))
        #     t += 1
        X_n = x_u_u[topNIndex]
        # print (x_u_u.shape)
        # print (X_n.shape)
            
        X_n = X_n[1:, :,: ]
        X_n = np.unique(X_n, axis=0)
        Y_n = np.zeros(X_n.shape[0])
        
        ### Train with RN and TP
        # print (x_p.shape)
        # print (X_n.shape)
        X = np.concatenate((x_p, X_n), axis=0)
        Y = np.concatenate((y_p, Y_n), axis=0)
        
        model  = self.classifier
        ## model.fit_transform(X, Y)
        
        if x_validate is None:
            model.fit_transform(X,Y,
                      epochs=self.IniEpochs,sample_weight=None)
        else:
            for i in range(self.NxtEpochs):
                model.fit_transform(X,Y,
                          epochs=1,sample_weight=None)
                self.validate(x_validate,i + self.IniEpochs)
    
    def predict(self,x):
        return self.classifier.predict_proba(x)
        
        
def SaveDCR(model,title = "DeepDCR"):
    import pickle
    savepath = join(settings.MODEL_DIR,"%s"%(title))
    if not exists(savepath):
        makedirs(savepath)
    model.classifier.save(savepath)
    with open(join(savepath,"validation.pkl"),"wb") as output:
        pickle.dump(model.result_valid, output)
    
def LoadDCR(model,title = "DeepDCR"):
    import pickle
    savepath = join(settings.MODEL_DIR,"%s"%(title))
    if not exists(savepath):
        return 
    model.classifier.load(savepath)
    with open(join(savepath,"validation.pkl"),"rb") as input:
        model.result_valid = pickle.load(input)

if __name__ == "__main__":
    fault_number = 1
    input_length = 12
    step = 12
    F_ratio=0.2
    P_ratio=0.4
    Reverse = False 
    
    # from TEP_Dataset import TEP_Windowed_DataSet
    # x_train,y_train,x_test,y_test=TEP_Windowed_DataSet(
    #     fault_number=fault_number,
    #     input_length=input_length,
    #     step=step,
    #     subset=0.1,
    #     ambigous=0.8)
    
    from TEP_Dataset import LoadDataSet,ModifyTrainingData
    x_train,y_train,x_test,y_test = LoadDataSet(join(settings.DATA_DIR,"IDV(%d).pkl"%(fault_number)))
    X,Y = ModifyTrainingData(x_train,y_train,4000,F_ratio,P_ratio)
    
    t_norm = 10000
    Xt = np.concatenate([x_test[:t_norm,:,:],x_test[-int(t_norm*F_ratio):,:,:]])
    Yt = np.concatenate([y_test[:t_norm,:],y_test[-int(t_norm*F_ratio):,:]])
    
    from LSTM_regression_0322 import LSTM_classifier
    
    
    model = DeepDCR(
        LSTM_classifier(18))
    
    if Reverse:
        model.fit(X,1-(Y[:,0]==True),Xt)
        y_pred=1-model.predict(Xt)
    else:
        model.fit(X,(Y[:,0]==True),Xt)
        y_pred=model.predict(Xt)
    
    title = "DeepDCR(%d)[F%d%%P%d%%]"%(fault_number,F_ratio*100,P_ratio*100)
    SaveDCR(model,title)
    
    
    import matplotlib
    matplotlib.use('Qt5Agg')
    from utils import TrainingPlot
    A,P,R,F = TrainingPlot(model.result_valid,Yt[:,1])
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.savefig(join(settings.FIGURE_DIR,"%s.png"%(title)))
    print("%.2f"%(A[-1]*100))
    print("%.2f"%(P[-1]*100))
    print("%.2f"%(R[-1]*100))
    print("%.2f"%(F[-1]*100))
    ### Plot
    # from plot_settings import plt
    # ax = plt.figure().add_subplot(111)
    # show_cases = 5
    # plt.plot(y_pred[:show_cases*int(960/12),1],label=r"$\it{\hat{Y}}$")
    # plt.plot(y_test[:show_cases*int(960/12),1],label=r"$\it{Y}$")
    # plt.title("DeepDCR LSTM TEP-IDV(%d).png"%(fault_number))
    # plt.xlabel("sample")
    # plt.ylabel("$\it{Y}$")
    # plt.legend()
    # plt.savefig(r"..\..\figures\DeepDCR LSTM TEP-IDV(%d).png"%(fault_number))
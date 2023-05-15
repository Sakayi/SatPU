# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:44:23 2023

@author: Song Keyu
"""


import numpy as np

TrainEpochs = 80
class Baseline:
    
    def __init__(self,
                 regression):
        
        self.regression     = regression
        self.result_valid   = None
    def predict(self, x):
        if self.regression is None:
            return None
        return self.regression.predict(x)
    
    def validate(self,x_valid,epoch_id):
        if self.result_valid is None:
            self.result_valid = np.zeros((TrainEpochs,x_valid.shape[0]))
        self.result_valid[epoch_id] = self.predict(x_valid)[:,1]   
    
    
    def fit(self,train_x,train_y,show = False,x_validate = None):
        self.label_weights = np.ones(shape=(train_y.shape[0],))
        self.label_weights = self.label_weights + train_y[:,0]
        self.label_weights *= len(self.label_weights)/self.label_weights.sum()
        
        if x_validate is None:
            self.regression.fit(train_x,train_y,
                      epochs=TrainEpochs,sample_weight=self.label_weights)
        else:
            for i in range(TrainEpochs):
                self.regression.fit(train_x,train_y,
                          epochs=1,sample_weight=self.label_weights)
                self.validate(x_validate,i)
    
    def __str__(self):
        name = "BaselineSupervised_FrameWork + " +  self.regression.__str__()  
        return name
    
    
def SaveBaseline(model,title = "Baseline"):
    import pickle
    import os
    savepath = "../../models/%s"%(title)
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    model.regression.save(savepath)
    with open(os.path.join(savepath,"validation.pkl"),"wb") as output:
        pickle.dump(model.result_valid, output)
    
def LoadBaseline(model,title = "Baseline"):
    import pickle
    import os
    savepath = "../../models/%s"%(title)
    if not os.path.exists(savepath):
        return 
    model.regression.load(savepath)
    with open(os.path.join(savepath,"validation.pkl"),"rb") as input:
        model.result_valid = pickle.load(input)

if __name__ == "__main__":
    fault_number = 1
    input_length = 12
    step = 12
    F_ratio=0.2
    P_ratio=0.4
    
    from TEP_Dataset import LoadDataSet,ModifyTrainingData
    x_train,y_train,x_test,y_test = LoadDataSet("../../data/IDV(%d).pkl"%(fault_number))
    X,Y = ModifyTrainingData(x_train,y_train,4000,F_ratio,P_ratio)
    
    t_norm = 10000
    Xt = np.concatenate([x_test[:t_norm,:,:],x_test[-int(t_norm*F_ratio):,:,:]])
    Yt = np.concatenate([y_test[:t_norm,:],y_test[-int(t_norm*F_ratio):,:]])
    from LSTM_regression import LSTM_regression
    
    modelbl = Baseline(
        LSTM_regression(18))
    modelbl.fit(X,Y,True,Xt)
    y_pred=modelbl.predict(Xt)
    
    title = "Baseline(%d)[F%d%%P%d%%]"%(fault_number,F_ratio*100,P_ratio*100)

    SaveBaseline(modelbl,title)
    
    import matplotlib
    matplotlib.use('Qt5Agg')
    from utils import TrainingPlot
    A,P,R,F = TrainingPlot(modelbl.result_valid,Yt[:,1])
    matplotlib.pyplot.title(title)
    matplotlib.pyplot.savefig("../../figures/training/%s.png"%(title))
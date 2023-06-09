# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 17:04:56 2022

@author: Song Keyu
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras import backend as K

import settings
from os.path import join,exists
from os import makedirs

def sharpen(x):
    return 0.5-0.5*np.cos(x*np.pi)

def negative_to_unlabeled(y_train):
    y_train_unlabeled = y_train
    y_train_unlabeled[y_train_unlabeled[:,1]==1.0]=0.5
    y_train_unlabeled[y_train_unlabeled[:,0]==0.0]=0.5
    return y_train_unlabeled




class SatPU:    
    
    def __init__(self,
                 regression,
                 momentum                        = 0.9,
                 filter_window                   = 5,
                 IniEpochs                       = 20, 
                 SatEpochs                       = 60, 
                 USE_PSEUDO_LABELING             = True,
                 USE_NONELINEAR_REWEIGHTING      = True,
                 USE_AMBIGUOUS_INITIALIZATION    = True,
                 USE_TEMPORAL_FILTER             = True):
        self.momentum       = momentum
        self.regression     = regression
        self.filter_window  = filter_window
        self.result_valid   = None
        self.IniEpochs      = IniEpochs
        self.SatEpochs      = SatEpochs

        self.USE_PSEUDO_LABELING                = USE_PSEUDO_LABELING
        self.USE_NONELINEAR_REWEIGHTING         = USE_NONELINEAR_REWEIGHTING
        self.USE_AMBIGUOUS_INITIALIZATION       = USE_AMBIGUOUS_INITIALIZATION
        self.USE_TEMPORAL_FILTER                = USE_TEMPORAL_FILTER
        
    def predict(self, x):
        if self.regression is None:
            return None
        return self.regression.predict(x)

    def pseudo_label(self,pred,epoch):
        sc = StandardScaler().fit_transform(np.array(pred[:,0]).reshape(-1,1))
        if self.USE_TEMPORAL_FILTER:
            from scipy.signal import medfilt
            sc = medfilt(sc.reshape(1,-1)[0],self.filter_window) 
        target = np.array(K.sigmoid(sc.reshape(-1,1)))
        
        if self.USE_NONELINEAR_REWEIGHTING:
            self.label_weights = (pred[:,1]-0.5)**2
            self.label_weights = 16*np.power(self.label_weights,2.0) + 0.0001
            self.label_weights *= len(self.label_weights)/np.sum(self.label_weights)
        else:
            self.label_weights = np.max(pred,axis=1) # Original SAT
        
        target = (target - np.min(target))/(np.max(target) - np.min(target))
        target = sharpen(target)
        target = np.concatenate((target,1-target),axis=1)
        self.soft_y = self.soft_y*(self.momentum) + target*(1-self.momentum)
        # self.soft_y = np.arctan(np.tan((self.soft_y-0.5)*np.pi)*(self.momentum) + 
        #                         np.tan((target-0.5)*np.pi)*(1-self.momentum)) / np.pi+0.5
        
    def plot(self,pred,title,L=400):
        import matplotlib
        import matplotlib.pyplot as plt
        matplotlib.use('Agg')
        x=np.linspace(0, self.soft_y.shape[0], self.soft_y.shape[0],False)
        plt.figure(figsize=(12,8))
        plt.scatter(x,self.label_weights/np.max(self.label_weights),label="weight",s=1)
        plt.scatter(x,pred[:,0],label="pred",s=1)
        plt.scatter(x,self.soft_y[:,0],label="pseudo",s=1)
        plt.title(title)
        plt.legend(loc="lower center")
        plt.savefig(join(settings.FIGURE_DIR,settings.TRAINING_PROCESS_SUBFOLDER,"%s.png"%(title)))
        plt.close()
        
    def save_intermediate(self,pred,file):
        import pickle
        with open(file,"wb") as output:
            pickle.dump(self.label_weights/np.max(self.label_weights), output)
            pickle.dump(pred[:,0], output)
            pickle.dump(self.soft_y[:,0], output)
        
    
    ### save validation result in training process as self.result_valid[epoch,sample]    
    def validate(self,x_valid,epoch_id):
        if self.result_valid is None:
            self.result_valid = np.zeros((self.IniEpochs+self.SatEpochs,x_valid.shape[0]))
        self.result_valid[epoch_id] = self.predict(x_valid)[:,1]   
        
        
        
    def fit(self,train_x,train_y,show = False,x_validate = None):
        
        self.label_weights = np.ones(shape=(train_y.shape[0],))
        self.label_weights = self.label_weights + train_y[:,0]
        self.label_weights *= len(self.label_weights)/self.label_weights.sum()
        
        if self.USE_AMBIGUOUS_INITIALIZATION:
            train_y = negative_to_unlabeled(train_y)
        
        if x_validate is None:
            self.regression.fit(train_x,train_y,
                      epochs=self.IniEpochs,sample_weight=self.label_weights)
        else:
            for i in range(self.IniEpochs):
                self.regression.fit(train_x,train_y,
                          epochs=1,sample_weight=self.label_weights)
                self.validate(x_validate,i)
                
        self.soft_y = np.array(train_y)
        
        for epoch in range(self.SatEpochs):
            y_pred = self.predict(train_x)
            
            if self.USE_PSEUDO_LABELING:
                self.pseudo_label(y_pred,epoch) 
            else:
                self.soft_y = self.soft_y*(self.momentum) + y_pred*(1-self.momentum)
                 
            if show:
                self.plot(y_pred,"%d"%(epoch))
                if epoch == 3:
                    self.save_intermediate(y_pred,join(settings.FIGURE_DIR,"epoch3.pkl"))
                print ("SatPU epoch %d"%(epoch))
            
            self.regression.fit(train_x,self.soft_y,
                      epochs=1,sample_weight=self.label_weights)
            
            if not x_validate is None:
                self.validate(x_validate,self.IniEpochs+epoch)
    
        
    
    def __str__(self):
        name = "SelfAdaptiveTraining_PUFrameWork + " +  self.regression.__str__()  
        return name
    
    
    def AblationCode(self):
        return  ("P" if self.USE_PSEUDO_LABELING  else "p") + \
                ("R" if self.USE_NONELINEAR_REWEIGHTING  else "r") + \
                ("A" if self.USE_AMBIGUOUS_INITIALIZATION  else "a") + \
                ("T" if self.USE_TEMPORAL_FILTER  else "t") +"_" + \
                ("T" if self.USE_PSEUDO_LABELING  else "F") + \
                ("T" if self.USE_NONELINEAR_REWEIGHTING  else "F") + \
                ("T" if self.USE_AMBIGUOUS_INITIALIZATION  else "F") + \
                ("T" if self.USE_TEMPORAL_FILTER  else "F")
#SatPU.ABLATION_CODE = UpdateAblationCode()
#print ("SatPU_"+SatPU.ABLATION_CODE)
    

def SaveSatPU(model,title = "SatPU",folder = settings.MODEL_DIR):
    import pickle
    savepath = join(folder,title)
    if not exists(savepath):
        makedirs(savepath)
    model.regression.save(savepath)
    with open(join(savepath,"validation.pkl"),"wb") as output:
        pickle.dump(model.result_valid, output)
    
def LoadSatPU(model,title = "SatPU",folder = settings.MODEL_DIR):
    import pickle
    savepath = join(folder,title)
    if not exists(savepath):
        raise FileNotFoundError(savepath)
    model.regression.load(savepath)
    with open(join(savepath,"validation.pkl"),"rb") as input:
        model.result_valid = pickle.load(input)

if __name__ == "__main__":
    fault_number = 1
    input_length = 12
    step = 12
    F_ratio=0.2
    P_ratio=0.4
    
    # from TEP_Dataset import TEP_Windowed_DataSet
    # x_train,y_train,x_test,y_test=TEP_Windowed_DataSet(
    #     fault_number=fault_number,
    #     input_length=input_length,
    #     step=step,
    #     subset=0.1,
    #     ambigous=0.0)
    
    from TEP_Dataset import LoadDataSet,ModifyTrainingData
    x_train,y_train,x_test,y_test = LoadDataSet(join(settings.DATA_DIR,"IDV(%d).pkl"%(fault_number)))
    X,Y = ModifyTrainingData(x_train,y_train,4000,F_ratio,P_ratio)
    
    t_norm = 10000
    Xt = np.concatenate([x_test[:t_norm,:,:],x_test[-int(t_norm*F_ratio):,:,:]])
    Yt = np.concatenate([y_test[:t_norm,:],y_test[-int(t_norm*F_ratio):,:]])
    
    # from LSTM_regression_0322 import LSTM_regression
    # model = SatPU(
    #     LSTM_regression(18),
    #     momentum=0.8)
    
    from MLP_regression import MLP_regression
    model = SatPU(
        MLP_regression(120),
        momentum=0.8)
    
    model.fit(X,Y,True,Xt)
    y_pred=model.predict(Xt)
    
    
    title = "SatPU_%s(%d)[F%d%%P%d%%]"%(model.AblationCode(),fault_number,F_ratio*100,P_ratio*100)
    SaveSatPU(model,title)
    
    
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
    # plt.title("SAT LSTM TEP-IDV(%d).png"%(fault_number))
    # plt.xlabel("sample")
    # plt.ylabel("$\it{Y}$")
    # plt.legend()
    # plt.savefig(r"..\..\figures\SAT LSTM TEP-IDV(%d)[F%.1f P%.1f].png"%(fault_number,F_ratio,P_ratio))
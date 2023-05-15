# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 17:44:36 2023

@author: Song Keyu
"""
import numpy as np


def TrainingPlot(result_mat,y_true,plot=True):
    '''
    example
    ------
    A,P,R,F = TrainingPlot(model.result_valid,Yt[:1])
    '''
    Epochs      = result_mat.shape[0]
    Accuracy    = np.zeros((Epochs,))
    Precision   = np.zeros((Epochs,))
    Recall      = np.zeros((Epochs,))
    F1          = np.zeros((Epochs,))
    for i in range(Epochs):
        Accuracy[i]  = np.sum((result_mat[i,:]>=0.5) ^ (y_true==0)) / len(y_true)
        Precision[i] = np.sum((result_mat[i,:]>=0.5) & (y_true==1)) / np.sum(result_mat[i,:]>=0.5)
        Recall[i]    = np.sum((result_mat[i,:]>=0.5) & (y_true==1)) / np.sum(y_true==1)
        F1[i]        = 2.0* Precision[i] * Recall[i] / (Precision[i] + Recall[i])
    
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(Accuracy   ,label="Accuracy")
        plt.plot(Precision  ,label="Precision")
        plt.plot(Recall     ,label="Recall")
        plt.plot(F1         ,label="F1")
        plt.xlabel("epoch")
        plt.ylim(0,1)
        plt.legend()
        
    return Accuracy,Precision,Recall,F1

def TrainingPlot_TF(result_mat,y_true,plot=True):
    '''
    example
    ------
    TN,TP,FN,FP = TrainingPlot_TF(model.result_valid,Yt[:,1])
    '''
    Epochs      = result_mat.shape[0]
    TN = np.zeros((Epochs,))
    TP = np.zeros((Epochs,))
    FN = np.zeros((Epochs,))
    FP = np.zeros((Epochs,))
    for i in range(Epochs):
        TN[i]  = np.sum((result_mat[i,:]<0.5)  & (y_true==0))
        TP[i]  = np.sum((result_mat[i,:]>=0.5) & (y_true==1))
        FN[i]  = np.sum((result_mat[i,:]<0.5)  & (y_true==1))
        FP[i]  = np.sum((result_mat[i,:]>=0.5) & (y_true==0))
        
    if plot:
        import matplotlib.pyplot as plt
        plt.plot(TN,label="TN")
        plt.plot(TP,label="TP")
        plt.plot(FN,label="FN")
        plt.plot(FP,label="FP")
        plt.xlabel("epoch")
        plt.ylim(0,1)
        plt.legend()
    return TN,TP,FN,FP


def GetTestY_TEP(fault_number=1,F_ratio=0.2,t_norm = 10000):
    from TEP_Dataset import LoadDataSet #,ModifyTrainingData
    x_train,y_train,x_test,y_test = LoadDataSet("../../data/IDV(%d).pkl"%(fault_number))
    
    # Xt = np.concatenate([x_test[:t_norm,:,:],x_test[-int(t_norm*F_ratio):,:,:]])
    Yt = np.concatenate([y_test[:t_norm,:],y_test[-int(t_norm*F_ratio):,:]])
    
    return Yt


    
def ModelTitle(name = "SelfAT",F_ratio=0.2,P_ratio=0.4,fault_number=1):
    return "%s(%d)[F%d%%P%d%%]"%(name,fault_number,int(F_ratio*100),int(P_ratio*100))

def LoadResults(folder):
    import os
    import pickle
    with open(os.path.join(r"../../models/",folder,"validation.pkl"),"rb") as input:
        result = pickle.load(input)
        return result
    return None

'''
Example
-------
Result = LoadResults(ModelTitle("DeepDCR",0.2,0.75,1))
Yt = GetTestY_TEP(1,0.2)
A,P,R,F = TrainingPlot(Result,Yt)
print("%.2f"%(A[-1]*100))
print("%.2f"%(P[-1]*100))
print("%.2f"%(R[-1]*100))
print("%.2f"%(F[-1]*100))
'''

def ComparativeExperiments(model_list,setting,Yt,Reverse):
    '''
    Example
    -------
    Yt = GetTestY_TEP(1,0.2)
    
    A,P,R,F = ComparativeExperiments(["Baseline","SelfAT","DeepDCR","DistPU"],"(1)[F20%P40%]",
                                     Yt[:,1],
                                     [False,False,False,True])
    A,P,R,F = ComparativeExperiments(["Baseline","SelfAT_PRAT_TTTT","DeepDCR","DistPU"],"(1)[F20%P40%]",
                                     Yt[:,0],
                                     [True,True,True,False])
    A,P,R,F = ComparativeExperiments(["Baseline","SelfAT_prat_FFFF",
                                      "SelfAT_PRat_TTFF","SelfAT_pRAt_FTTF",
                                      "SelfAT_PrAt_TFTF","SelfAT_pRat_FTFF",
                                      "SelfAT_Prat_TFFF","SelfAT_prAt_FFTF",
                                      "SelfAT_PRAt_TTTF","SelfAT_PRAT_TTTT",],
                                     "(1)[F20%P40%]",
                                     Yt[:,0],
                                     [True,True,True,True,True,True,True,True,True,True])
    
    ### NEW
    A,P,R,F = ComparativeExperiments(["Baseline","SatPU_PRAT_TTTT5","DeepDCR"],"(TEP-1)[F20%P40%]",
                                     Yt[:,0],
                                     [True,True,True])
    
    A,P,R,F = ComparativeExperiments(["Baseline","SatPU_PRAT_TTTT5","DeepDCR","DistPU"],"(TEP-1)[F20%P40%]",
                                     Yt[:,0],
                                     [True,True,True,False])
    
    A,P,R,F = ComparativeExperiments(["Baseline","SatPU_PRAT_TTTT5","DeepDCR","DistPU"],"(TEP-1)[F20%P40%]",
                                     Yt[:,1],
                                     [False,False,False,True])
    
    from DAMADICS_Dataset import TrainingSet1109,TestSet1117
    X,Y,scaler = TrainingSet1109(1-0.4,length=12,step=1,actuator="Actuator3")
    Xt,Yt = TestSet1117(scaler,length=12,step=1,actuator="Actuator3")
    
    A,P,R,F = ComparativeExperiments(["Baseline","SatPU_PRAT_TTTT","DeepDCR"],
                                     "(DAM-09-17#3)[F20%P40%]",
                                     Yt,
                                     [True,True,True])
    A,P,R,F = ComparativeExperiments(["Baseline","SatPU_PRAT_TTTT","DeepDCR","DistPU"],
                                     "(DAM-09-17#3)[F20%P20%]",
                                     Yt,
                                     [True,True,True,False])
    
    A,P,R,F = ComparativeExperiments(["Baseline","SatPU_PRAT_TTTT","DeepDCR","DistPU"],
                                     "(DAM-09-17#3)[F20%P20%]",
                                     1-Yt,
                                     [False,False,False,True])
    '''
    Results=[]
    As,Ps,Rs,Fs = [],[],[],[]
    for i in range(len(model_list)):
        Results.append(LoadResults("%s%s"%(model_list[i],setting)))
        A,P,R,F = TrainingPlot(1-Results[-1] if Reverse[i] else Results[-1],Yt,False)
        As.append(A),Ps.append(P),Rs.append(R),Fs.append(F)
        print (model_list[i])
        print ("%.2f\t%.2f\t%.2f\t%.2f\t"%(A[-1]*100,P[-1]*100,R[-1]*100,F[-1]*100))
    return As,Ps,Rs,Fs
        
def GroupBarPlot(DF,width = 0.2):
    import matplotlib.pyplot as plt
    tick_label = ["%d%%"%(np.ceil(100*DF.index[i])) for i in range(DF.shape[0])]
    for i in range(DF.shape[1]):
        plt.bar(np.array(range(DF.shape[0]))+width*i,
                DF.iloc[:,i],
                width=width,
                label=DF.columns[i],)
    # plt.legend()
    plt.xticks(np.array(range(DF.shape[0]))+width*(DF.shape[1]*0.5-0.5),labels=tick_label)
    # plt.xticklabels(tick_label)
    
def UnlabeledRatioPlot(model_list,dataset,F_ratio,P_ratio_list,Yt,Reverse,model_names,repeated=1,plot="plot"):
    '''
    example
    -------
    Yt = GetTestY_TEP(1,0.2)
    UnlabeledRatioPlot(["Baseline","SatPU_PRAT_TTTT","DeepDCR","DistPU"],
                       "TEP-1", 0.2,
                       [0.8,0.75,0.6,0.4,0.2],
                       Yt[:,1],
                       [False,False,False,True],
                       ["Baseline","SatPU","DeepDCR","Dist-PU"],
                       1,
                       "plot")
    
    
    from DAMADICS_Dataset import TrainingSet1109,TestSet1117
    X,Y,scaler = TrainingSet1109(1-0.4,length=12,step=1,actuator="Actuator3")
    Xt,Yt = TestSet1117(scaler,length=12,step=1,actuator="Actuator3")
    UnlabeledRatioPlot(["Baseline","SatPU_PRAT_TTTT","DeepDCR"],
                       "DAM-09-17#3", 0.2,
                       [0.8,0.75,0.6,0.4,0.2],
                       1-Yt,
                       [False,False,False],
                       ["Baseline","SatPU","DeepDCR"],
                       30,
                       "groupbar")
    UnlabeledRatioPlot(["Baseline","SatPU_PRAT_TTTT","DeepDCR","DistPU"],
                       "DAM-09-17#3", 0.2,
                       [0.8,0.75,0.6,0.4,0.2],
                       1-Yt,
                       [False,False,False,True],
                       ["Baseline","SatPU","DeepDCR","Dist-PU"],
                       20,
                       "groupbar")
    
    '''
    import pandas as pd
    Acc = pd.DataFrame(columns=model_list)
    Pre = pd.DataFrame(columns=model_list)
    Rec = pd.DataFrame(columns=model_list)
    F1s = pd.DataFrame(columns=model_list)
    for P_ratio in P_ratio_list:
        index = 1 - P_ratio
        if index <= F_ratio and "TEP" in dataset:
            index = 0.0
        As = [0] * len(model_list)
        Ps = [0] * len(model_list)
        Rs = [0] * len(model_list)
        Fs = [0] * len(model_list)
        for i in range(len(model_list)):
            if repeated > 1:
                As[i],Ps[i],Rs[i],Fs[i] = [],[],[],[]
                for rp in range(1,repeated+1,1):
                    result = LoadResults("%s(%s)[F%d%%P%d%%]%d"%(model_list[i],dataset,F_ratio*100,P_ratio*100,rp))
                    A,P,R,F = TrainingPlot(1-result if Reverse[i] else result,Yt,False)
                    As[i].append(A[-1])
                    Ps[i].append(P[-1])
                    Rs[i].append(R[-1])
                    Fs[i].append(F[-1])
                # As[i] = np.median(As[i])
                # Ps[i] = np.median(Ps[i])
                # Rs[i] = np.median(Rs[i])
                # Fs[i] = np.median(Fs[i])
                As[i] = np.mean(As[i])
                Ps[i] = np.mean(Ps[i])
                Rs[i] = np.mean(Rs[i])
                Fs[i] = np.mean(Fs[i])
                # As[i] = As[i][np.argmax(Fs[i])]
                # Ps[i] = Ps[i][np.argmax(Fs[i])]
                # Rs[i] = Rs[i][np.argmax(Fs[i])]
                # Fs[i] = np.max(Fs[i])
            else:
                result = LoadResults("%s(%s)[F%d%%P%d%%]"%(model_list[i],dataset,F_ratio*100,P_ratio*100))
                A,P,R,F = TrainingPlot(1-result if Reverse[i] else result,Yt,False)
                As[i],Ps[i],Rs[i],Fs[i] = A[-1],P[-1],R[-1],F[-1]
        
        Acc.loc[index]=As
        Pre.loc[index]=Ps
        Rec.loc[index]=Rs
        F1s.loc[index]=Fs
        
    import matplotlib.pyplot as plt
    fig = plt.figure()
    fig.tight_layout(pad=25.0)
    for DF,subplot,name in [(Acc,221,"Accuracy"),
                            (F1s,222,"F1 Score"),
                            (Pre,223,"Precision"),
                            (Rec,224,"Recall")]:
        plt.subplot(subplot)
        if plot == "plot":
            plt.plot(DF)
        elif plot == "groupbar":
            GroupBarPlot(DF)
        print (name)
        print (DF)
        plt.ylim(0,1.0)
        plt.xlabel("u")
        plt.title(name)
    plt.legend(model_names,bbox_to_anchor=(0,1.25),loc='upper right')
    

    
def ModelF1Boxplot(model,dataset,F_ratio,P_ratio_list,Yt,reverse,reps,model_name):
    '''
    example
    -------
    from DAMADICS_Dataset import TrainingSet1109,TestSet1117
    X,Y,scaler = TrainingSet1109(1-0.4,length=12,step=1,actuator="Actuator3")
    Xt,Yt = TestSet1117(scaler,length=12,step=1,actuator="Actuator3")
    ModelF1Boxplot("DeepDCR","DAM-09-17#3",0.2,[0.8,0.75,0.6,0.4,0.2],
                   1-Yt,False,range(1,11,1),"DeepDCR")
    ModelF1Boxplot("SatPU_PRAT_TTTT","DAM-09-17#3",0.2,[0.8,0.75,0.6,0.4,0.2],
                   1-Yt,False,range(1,11,1),"SatPU")
    
    
    '''
    import pandas as pd
    F1s = pd.DataFrame([],columns=reps)
    for P_ratio in P_ratio_list:
        index = int((1 - P_ratio)*100)
        # if index <= F_ratio:
        #     index = 0.0
        F1s.loc[index]=[0.0]*len(reps)
        for rp in reps:
            result = LoadResults("%s(%s)[F%d%%P%d%%]%d"%(model,dataset,F_ratio*100,P_ratio*100,rp))
            A,P,R,F = TrainingPlot(1-result if reverse else result,Yt,False)
            F1s.loc[index][rp]=F[-1]
        
    import matplotlib.pyplot as plt
    plt.figure()
    plt.boxplot(F1s.T,positions=F1s.index,widths=4)
    plt.title(model_name)
    plt.ylim(0,1)
    plt.xlim(F1s.index[0]-10,F1s.index[-1]+10)
    plt.xlabel("u%")
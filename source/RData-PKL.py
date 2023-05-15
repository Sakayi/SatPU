# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 22:25:49 2022

@author: Song Keyu

Run in environment R-py-env (python3.7+pyreadr)
"""

# import rpy2.robjects as robjects
# import numpy as np
import pandas as pd
import pyreadr
import pickle
#robjects.r['load'](r"C:\Users\admin\Desktop\宋克禹\2022\变点检测\dataverse_files\TEP_FaultFree_Testing.Rdata")

import settings
from os.path import join

PklFolder = r"C:\Users\admin\Desktop\宋克禹\2022\智能电厂\数据\数据包"
def DataFrameToPKL(DF:pd.DataFrame,file_name:str):
    with open(join(settings.TEP_DATA_DIR,file_name),"wb") as output:
        pickle.dump(DF, output)



for FileName in  ["TEP_FaultFree_Testing","TEP_FaultFree_Training","TEP_Faulty_Testing","TEP_Faulty_Training"]:
    OD = pyreadr.read_r(join(settings.TEP_DATA_DIR,"%s.Rdata"%(FileName)))
    DF = pd.DataFrame(list(OD.values())[0])
    DataFrameToPKL(DF,FileName+".pkl")

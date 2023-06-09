# Set up Python environment (Anaconda)
From yaml file
```bash
conda create --name _KERAS_GPU_ --file SatPU.yaml
```

Or build from scratch:
```bash
conda create --name _KERAS_GPU_ keras-gpu=2.6.0 numpy=1.21.5 python=3.9
conda activate _KERAS_GPU_
conda install pandas=1.5.3
conda install scikit-learn=1.2.2
pip install matplotlib==3.7.1
pip install pyQt5==5.15.9
pip install pyreadr==0.4.7
```


# Reproducing results in the paper

## Download benchmark datasets
### TEP
The Tennessee Eastman Process(TEP) extended dataset is available online:
<https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/6C3JR1>

Download rdata files:

TEP_FaultFree_Testing.RData:    <https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/6C3JR1/Q4NOFD&version=1.0>

TEP_FaultFree_Training.RData:   <https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/6C3JR1/TMBUVJ&version=1.0>

TEP_Faulty_Testing.RData:   <https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/6C3JR1/IL0XP0&version=1.0>

TEP_Faulty_Training.RData:  <https://dataverse.harvard.edu/file.xhtml?persistentId=doi:10.7910/DVN/6C3JR1/6QOWUU&version=1.0>

Transform RData files into pkl files (Necessary step)
```bash
cd source
python RData-PKL.py
```

### DAMADICS
DAMADICS dataset is available online:
<https://iair.mchtr.pw.edu.pl/Damadics>

Download zip files(4 parts):

<https://iair.mchtr.pw.edu.pl/content/download/163/817/file/Lublin_all_data_part1.zip>

<https://iair.mchtr.pw.edu.pl/content/download/163/817/file/Lublin_all_data_part2.zip>

<https://iair.mchtr.pw.edu.pl/content/download/163/817/file/Lublin_all_data_part3.zip>

<https://iair.mchtr.pw.edu.pl/content/download/163/817/file/Lublin_all_data_part4.zip>

Then extract the txt files to data folder : data\DAMADICS


## Conduct experiments and save results
### TEP IDV-1 
```bash
cd source
python RunExperiment.py --dataset_name TEP --step 12 --method SatPU    --P_ratio 0.8
python RunExperiment.py --dataset_name TEP --step 12 --method DeepDCR  --P_ratio 0.8
python RunExperiment.py --dataset_name TEP --step 12 --method Baseline --P_ratio 0.8
python RunExperiment.py --dataset_name TEP --step 12 --method SatPU    --P_ratio 0.75
python RunExperiment.py --dataset_name TEP --step 12 --method DeepDCR  --P_ratio 0.75
python RunExperiment.py --dataset_name TEP --step 12 --method Baseline --P_ratio 0.75
python RunExperiment.py --dataset_name TEP --step 12 --method SatPU    --P_ratio 0.6
python RunExperiment.py --dataset_name TEP --step 12 --method DeepDCR  --P_ratio 0.6
python RunExperiment.py --dataset_name TEP --step 12 --method Baseline --P_ratio 0.6
python RunExperiment.py --dataset_name TEP --step 12 --method SatPU    --P_ratio 0.4
python RunExperiment.py --dataset_name TEP --step 12 --method DeepDCR  --P_ratio 0.4
python RunExperiment.py --dataset_name TEP --step 12 --method Baseline --P_ratio 0.4
python RunExperiment.py --dataset_name TEP --step 12 --method SatPU    --P_ratio 0.2
python RunExperiment.py --dataset_name TEP --step 12 --method DeepDCR  --P_ratio 0.2
python RunExperiment.py --dataset_name TEP --step 12 --method Baseline --P_ratio 0.2
```
Or run .bat file:
```bash
call ConductExperiments-TEP.bat
```

### DAMADICS Actuator3
```bash
cd source
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method SatPU    --P_ratio 0.8  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method DeepDCR  --P_ratio 0.8  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method Baseline --P_ratio 0.8  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method SatPU    --P_ratio 0.75 --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method DeepDCR  --P_ratio 0.75 --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method Baseline --P_ratio 0.75 --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method SatPU    --P_ratio 0.6  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method DeepDCR  --P_ratio 0.6  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method Baseline --P_ratio 0.6  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method SatPU    --P_ratio 0.4  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method DeepDCR  --P_ratio 0.4  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method Baseline --P_ratio 0.4  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method SatPU    --P_ratio 0.2  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method DeepDCR  --P_ratio 0.2  --caseid %%I
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name DAMADICS --actuator Actuator3 --step 1 --method Baseline --P_ratio 0.2  --caseid %%I
```
Or run .bat file:
```bash
call ConductExperiments-DAMADICS.bat
```

### DistPU
Dist-PU runs in pytorch environment, please refer to another git repo:

<https://github.com/Sakayi/Dist-PU>

### Ablation study
```bash
cd source
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 0 --TEMPORAL_FILTER 0 --step 12
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 0 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 0 --TEMPORAL_FILTER 0 --step 12
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 0 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 1 --step 12
```
Or run .bat file:
```bash
call ConductExperiments-AblationStudy.bat
```

## Organize experiment results

See functions and examples in source\utils.py
<!-- TODO show results -->
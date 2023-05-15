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
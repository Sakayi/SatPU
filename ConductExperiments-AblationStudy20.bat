cd source
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name TEP --method Baseline --step 12 --caseid %%I --load_saved True

for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12 --caseid %%I --load_saved True
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 0 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12 --caseid %%I --load_saved True
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 0 --TEMPORAL_FILTER 0 --step 12 --caseid %%I --load_saved True

for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 0 --TEMPORAL_FILTER 0 --step 12 --caseid %%I --load_saved True
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 0 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 0 --TEMPORAL_FILTER 0 --step 12 --caseid %%I --load_saved True
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 0 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12 --caseid %%I --load_saved True

for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12 --caseid %%I --load_saved True
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 0 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 0 --TEMPORAL_FILTER 0 --step 12 --caseid %%I --load_saved True
for %%I in (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20) do python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 1 --step 12 --caseid %%I --load_saved True

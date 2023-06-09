cd source
python RunExperiment.py --dataset_name TEP --method Baseline --step 12 --P_ratio 0.4 --load_saved True
 
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12 --P_ratio 0.4 --load_saved True
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 0 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12 --P_ratio 0.4 --load_saved True
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 0 --TEMPORAL_FILTER 0 --step 12 --P_ratio 0.4 --load_saved True

python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 0 --TEMPORAL_FILTER 0 --step 12 --P_ratio 0.4 --load_saved True
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 0 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 0 --TEMPORAL_FILTER 0 --step 12 --P_ratio 0.4 --load_saved True
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 0 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12 --P_ratio 0.4 --load_saved True

python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 0 --step 12 --P_ratio 0.4 --load_saved True
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 0 --NONELINEAR_REWEIGHTING 0 --AMBIGUOUS_INITIALIZATION 0 --TEMPORAL_FILTER 0 --step 12 --P_ratio 0.4 --load_saved True
python RunExperiment.py --dataset_name TEP --PSEUDO_LABELING 1 --NONELINEAR_REWEIGHTING 1 --AMBIGUOUS_INITIALIZATION 1 --TEMPORAL_FILTER 1 --step 12 --P_ratio 0.4 --load_saved True
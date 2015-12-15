The following are descriptions of each individual file: 

- dataprep_script.py 
    - Apply Gaussian filter to each run and save each smoothed run into data as 'smoothed_run_[index of run]'
- mask_generating.py 
    - Using 'smoothed_runs.npy' in data folder, create 50k, 17k, and 9k masks. This then saves these masks ('sm_mask55468.npy', 'sm_mask17887.npy', 'sm_mask9051.npy') into the 'brain_mask' folder. 
- filter_script.py 
    - Using the masks generated from 'mask_generating.py,' save the subset of the data into the data folder ('masked_data_50k.npy', 'masked_data_17k.npy', 'masked_data_9k.npy'). It also saves visuals of this new filtered data into the figures folder ('smoothed3mm.jpeg','smoothed3mm_50k.jpeg', 'smoothed3mm_17k.jpeg', 'smoothed3mm_9k.jpeg'). 
- data_filtering.py 
    - Using the the 50k subset data ('masked_data_50k.npy') created in 'filter_script.py', this does more filtering and cleaning and saves the resulting data into the data folder as 'filtered_data.npy'. We also create 'data_filtering_on_smoothed_data.jpg' and save this to the 'figure' folder. 

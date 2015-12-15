# Project Beta
## UC Berkeley's Statistics 159/259
### Project Group Beta, Fall Term 2015 

[![Build Status](https://travis-ci.org/berkeley-stat159/project-beta.svg?branch=master)](https://travis-ci.org/berkeley-stat159/project-beta) 
[![Coverage Status](https://coveralls.io/repos/berkeley-stat159/project-beta/badge.svg?branch=master&service=github)](https://coveralls.io/github/berkeley-stat159/project-beta?branch=master)

_**Topic:**_ [Modeling of Semantic Representation in the Brain Using fMRI Response] (https://openfmri.org/dataset/ds000113)
_**Group members:**_ Agrawal Raj, Dong Yucheng, Mo Cindy, Sinha Rishi & Wang, Yuan

## Installation
1. Install python dependencies with pip: `pip install -r requirements.txt`
2. Clone the project repository: `git clone https://github.com/berkeley-stat159/project-beta.git'
3. cd to the project-beta directory 


## Roadmap

#### Repository Navigation Overview 
1. make data -  downloads the data for analysis
2. make validate - ensure the data is not corrupted
3. make coverage - runs coverage tests and generates the Travis coverage report
4. make test - runs all of the tests for scripts 
5. make Preprocess data - preprocess the data
6. make Preprocess Description - Preprocess Description data
7. make analysis - generate figures and results
8. make report - build final report

#### Data
- `make data_download`: Download a 5GB ds013-subject12 data for our analysis from Googledrive. It is a compressed file that contains 8 bold_dico.nii.gz from 8runs. We then rename it with the correct name. Data is then uncompressed. (~ 15-20 min)

#### Validate
- `make validate`: Validates the ds013-subject12 data by checking the right named file is present and has the correct hash. (~ 1 min)

#### Tests 
- `make test`: Runs all the tests for this project repository. The test functions are located in code/utils/tests (~ 2 min)

#### Coverage 
- `make coverage`: Runs the coverage tests and then generates a coverage report. (~ 2 min)

#### Preprocess data
- `make preprocess_data`: Runs all the scripts that are used for preprocess data analysis. (~ 30 - 60 min)

#### Preprocess Description
- `make preprocess_description`: Run all the scripts that are used for preprocess description. (~ 10 min)

#### Analysis
- `make analysis`: Runs all the scripts that are used for Analysis including Ridge Regression Analysis and Neural Network Analysis (~ 1 - 2 hr)

#### Report
- `make paper_report`: Build final report (~ 1 min)

## Discussion on Reproducibility 
All files mentioned below are located in the /code/ directory. 
The files are grouped by their functionalities. 

### 1. Disk Usage
- Note: All data and figures will occupy around 22GB.
	

### 2. Data Preprocessing 
- data_loading_script.py 
    - This does some basic EDA on the raw data (as done in HW2). We look at RMS and standard deviation spreads on voxel time courses. It saves these figures as 'std_plt[index of run].png' and 'rms_outliers[index of run].png'
- dataprep_script.py 
    - Apply Gaussian filter to each run and save each smoothed run into data as 'smoothed_run_[index of run]'
- mask_generating.py 
    - Using 'smoothed_runs.npy' in data folder, create 50k, 17k, and 9k masks. This then saves these masks ('sm_mask55468.npy', 'sm_mask17887.npy', 'sm_mask9051.npy') into the 'brain_mask' folder. 
- filter_script.py 
    - Using the masks generated from 'mask_generating.py,' save the subset of the data into the data folder ('masked_data_50k.npy', 'masked_data_17k.npy', 'masked_data_9k.npy'). It also saves visuals of this new filtered data into the figures folder ('smoothed3mm.jpeg','smoothed3mm_50k.jpeg', 'smoothed3mm_17k.jpeg', 'smoothed3mm_9k.jpeg'). 
- data_filtering.py 
    - Using the the 50k subset data ('masked_data_50k.npy') created in 'filter_script.py', this does more filtering and cleaning and saves the resulting data into the data folder as 'filtered_data.npy'. We also create 'data_filtering_on_smoothed_data.jpg' and save this to the 'figure' folder. 

### 3. Preprocess Description
- dataclean.py 
	- Cleaning up random words in translated movie description and tagging every word in the description with nltk WordNet labels.
- gen_design_matrix.py 
	- take in description in every time window and tranform it into a design matrix for modeling

### 4. Analysis 
- ridge regression 
	- description_modeling_ridge_regression.py 
		-  Uses ridge regression to simultaneously model all BOLD response time courses given the design matrix
		-  Determines a optimal ridge parameter with 10 fold cross validation
		-  Evaluates regression performance by computing correlationship coefficient between prediciton and the actual response
- k-nearest neighbors 
	- scenes_pred.py 
		- Uses KNN to try and predict what scene occured at a time point based on the brain image at that time point 
- neural network 
	- nn.py 
		- Creates a neural network using cross-entropy error and stochastic gradient descent to predict presence/absence of common objects in movie based on voxel responses

## Team Members  
- Raj Agrawal ([`raj4`](https://github.com/raj4))
- Steve Yucheng Dong ([`yuchengdong`](https://github.com/yuchengdong))
- Cindy Mo ([`cxmo`](https://github.com/cxmo))
- Rishi Sinha ([`rishizsinha`](https://github.com/rishizsinha))
- Aria Yuan Wang ([`ariaaay`](https://github.com/ariaaay))


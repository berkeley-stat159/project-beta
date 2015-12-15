# Project Beta
## UC Berkeley's Statistics 159/259
### Project Group Beta, Fall Term 2015 

[![Build Status](https://travis-ci.org/berkeley-stat159/project-beta.svg?branch=master)](https://travis-ci.org/berkeley-stat159/project-beta) 
[![Coverage Status](https://coveralls.io/repos/berkeley-stat159/project-beta/badge.svg?branch=master&service=github)](https://coveralls.io/github/berkeley-stat159/project-beta?branch=master)

_**Topic:**_ [Modeling of Semantic Representation in the Brain Using fMRI Response] (https://openfmri.org/dataset/ds000113)
_**Group members:**_ Agrawal Raj, Dong Yucheng, Mo Cindy, Sinha Rishi & Wang, Yuan


## Roadmap

#### Repository Navigation Overview 
1. make data -  downloads the data for analysis
2. make validate - ensure the data is not corrupted
3. make coverage - runs coverage tests and generates the Travis coverage report
4. make test - runs all of the tests for scripts 
5. make eda - generate figures from exploratory analysis
6. make analysis - generate figures and results
7. make report - build final report

#### Data
- `make data_process`: Download a 5GB ds013-subject12 data for our analysis from Googledrive. It is a compressed file that contains 8 bold_dico.nii.gz from 8runs. We then rename it with the correct name. Data is then uncompressed.

#### Validate
- `make validate`: Validates the ds013-subject12 data by checking the right named file is present and has the correct hash. 

#### Tests 
- `make test`: Runs all the tests for this project repository. The test functions are located in code/utils/tests

#### Coverage 
- `make coverage`: Runs the coverage tests and then generates a coverage report.

#### EDA
- `make eda`: Runs all the scripts that are used for exploratory data analysis.

#### Analysis
- `make analysis`: Runs all the scripts that are used for Analysis including Ridge Regression Analysis and Neural Network Analysis

#### Report
- `make report`: Build final report

## Discussion on Reproducibility 
All files mentioned below are located in the /code/ directory. 
The files are grouped by their functionalities. 

### 1. Download Data 
- data_loading_script.py 

### 2. Data Preprocessing 
- dataprep_script.py 
- mask_generating.py 
- filter_script.py 
- mask_generating.py 

### 3. Preprocess Description
- dataclean.py 
- gen_design_matrix.py 

### 4. Analysis 
- ridge regression 
	- description_modling_ridge_regression.py 
- k-nearest neighbors 
	- scenes_pred.py 
- neural network 
	- nn.py 

## Team Members  
- Raj Agrawal ([([`raj4`](https://github.com/raj4))
- Steve Yucheng Dong ([([`yuchengdong`](https://github.com/yuchengdong))
- Cindy Mo ([([`cxmo`](https://github.com/cxmo))
- Rishi Sinha ([([`rishizsinha`](https://github.com/rishizsinha))
- Aria Yuan Wang ([([`ariaaay`](https://github.com/ariaaay))


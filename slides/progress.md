% Project beta Progress Report
% Yuan Wang (Aria), Cindy Mo, Yucheng Dong (Steve), Rishi Sinha, Raj Agrawal
% November 12, 2015

# Background

## The Paper

- from OpenFMRI.org
- ds000113

## The Data

- 2 hour in total for 1 subject 
- 8 runs 
- 15 mins each run
- Format: 4D volumetric images (160x160x36)in NIfTI format
- TR: 2s
- Sequence: T2*-weighted gradient-echo EPI sequence (1.4 mm isotropic voxel size). 
- IMPORTANT NOTES: These images have partial brain coverage — centered on the auditory cortices
 in both brain hemispheres and include frontal and posterior portions of the 
 brain. There is no coverage for the upper portion of the brain (e.g. large 
 parts of motor and somato-sensory cortices).”

- Subject we downloaded: 004, (014, 015)

# ALL STEPS
## 1. Download the data and separate them into sessions (1 subjects all session ~15G) (DONE)
## 2. Data Preprocessing (WEEK 1) (Raj, Cindy, Steve)
### 2.1 Simple plots, summary statistics
- plot the subject 1 data for single run (run 001-run 008); [plot it as activation (y-axis) by time(x-axis), refer to hw2]
- check for outliers, trends in the data; clean them; (refer to hw2)

### 2.2 Data cleaning - Depends on how the data look like
- motion correction?

### 2.3 detrend, align runs
- compare across runs for all runs, and align run 001 to run 007 for modeling training data set; and run 008 for validation data set;

## 3. Construction of design matrix: movie description analysis (WEEK 1)(Aria, Rishi)
### 3.1 text cleaning(could skip a,b if c can be done directly)
- Take out German specific words, annotation, irrelevant words(“on”, “the”, “a”, “is”, “”) (Keep people’s names and adjective for now)
- eliminate case differences
- extract object/action/adjective words from the movie description

### 3.2 Tag description with WordNet labels (in order to obtain the tree structure of words in English)
### 3.3 construct a wordNet tree
### 3.4 construct a design matrix

## 4. Voxel modeling using ridge regression model (+cross validation using different parameter) (WEEK 2+3) (Everyone)
- similar to single voxel modeling that is talked about in class(day 15)

## 5. Prediction and visualize results (WEEK 3) (Cindy)


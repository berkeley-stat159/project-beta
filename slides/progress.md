% Project Beta Progress Report
% Yuan Wang (Aria), Cindy Mo, Yucheng Dong (Steve), Rishi Sinha, Raj Agrawal
% November 12, 2015

# Background

## Data Description
- high-resolution fMRI dataset 
  - 20 participants 
- audio description in German with start and end times
- dsnum: ds000113 

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

# Completed Steps 

## Data Preprocessing 
- Translated German audio description into English 
- Removed English stopwords (common words that hold no meaning) 
  from audio descriptions 
- Saved outlier data indices 

## WordNet 
- Tagged descriptions using WordNet labels and store in JSON file 
- Separate each English description into 2 second intervals 

## Initial Analysis 

# Future Plan 

## Construct a Design Matrix 

## Voxel Modeling Using Ridge Regression Model 
### Cross-validation 
- 70% of the data used as training 
- 30% used as test data 

## Two Kinds of Modeling
### Scenes Modeling 
- Setting where scenes take place
- ~ 50 seconds 
### Detailed Description Modeling 
- Divide words in each time window 
- Model each of the word in the description 
### Deduce from Brain Scan to Word Category 

## Problems 
- Learning Github flow
  - Learn to "unpush" something 
- Understanding fMRI data 
- Data manipulation to fit the question we're trying to answer 

# Process 

## Issues as Team 
- Finding right time to meet

## Class Feedback 
### Useful: 
- Code exercises and solutions 
- Github 
### Could Improve: 
- Lack of notes 
- More emphasis on fMRI data 
### Areas of Review: 
- ANOVA 
- Regression modeling 


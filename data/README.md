## Downloading Processing and Validating the Data

The Makefile contains recipes to Download, Process and Validate the data required for our analysis and tests.   

- `make`: This command would do all the following command together.

- `make download`: Download a 5GB ds013-subject12 data for our analysis from Googledrive. It is a compressed file that contains 8 bold_dico.nii.gz from 8runs. We then rename it with the correct name.

- `make process`: Unzip the file.

- `make validate`: Validates the ds013-subject12 data by checking the right named file is present and has the correct hash. 

- `make test`: Test get hash functions associated with validating data on both naming obtaining and checking file hashes. 

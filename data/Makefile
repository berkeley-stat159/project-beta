Download:
	wget http://openfmri.s3.amazonaws.com/tarballs/ds113_sub004.tgz

Unzip:	
	tar -zxvf ds113_sub004.tgz 
	cd sub004 
	find . -type f -name 'bold_dico.nii.gz' -exec gunzip -k "{}" \;

validate:
	python data.py

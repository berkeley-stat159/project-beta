all : Download Processing Validate

Download:
	wget -c https://www.googledrive.com/host/0Bz7lWLS0atxsbXZpT056ZnJnd1U
	wget -c https://www.googledrive.com/host/0B2Npp51nehlCSE5MYUo0NTVHS0U
	wget -c https://www.googledrive.com/host/0B2Npp51nehlCNDZ6QmhWYUVxMGM
	
Processing:
	mv '0Bz7lWLS0atxsbXZpT056ZnJnd1U' 'ds113_sub012.tgz'
	tar -zxvf ds113_sub012.tgz
	mv '0B2Npp51nehlCSE5MYUo0NTVHS0U' 'BOLD_est_masked34589.npy'
	mv '0B2Npp51nehlCNDZ6QmhWYUVxMGM' 'BOLD_val_masked34589.npy' 

Validate:
	python data.py


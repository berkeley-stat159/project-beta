all : Download Unzip validate

Download:
	wget -c https://www.googledrive.com/host/0Bz7lWLS0atxsalZSQWc1NXV2ajg
	wget -c https://drive.google.com/file/d/0B2Npp51nehlCSE5MYUo0NTVHS0U
	wget -c https://drive.google.com/file/d/0B2Npp51nehlCNDZ6QmhWYUVxMGM
Unzip:
	mv '0Bz7lWLS0atxsalZSQWc1NXV2ajg' 'ds113_sub004.tgz'
	tar -zxvf ds113_sub004.tgz 
	mv ‘0B2Npp51nehlCSE5MYUo0NTVHS0U’ ‘BOLD_est_masked34589.npy’
	mv ‘0B2Npp51nehlCNDZ6QmhWYUVxMGM’ ‘BOLD_val_masked34589.npy’

validate:
	python data.py


	
all : download process validate test

download:
	wget -c https://www.googledrive.com/host/0Bz7lWLS0atxsbXZpT056ZnJnd1U
	mv '0Bz7lWLS0atxsbXZpT056ZnJnd1U' 'ds113_sub012.tgz'
	wget -c https://www.googledrive.com/host/0BxlqqubRo4V3WTVkSXVNTktuLW8 
	mv '0BxlqqubRo4V3WTVkSXVNTktuLW8' 'smoothed_data.npy'

	
process:
	tar -zxvf ds113_sub012.tgz
	
validate:
	python data.py

test:
	nosetests tests 


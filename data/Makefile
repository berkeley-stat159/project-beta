all : download process validate test

download:
	wget -c https://www.googledrive.com/host/0Bz7lWLS0atxsbXZpT056ZnJnd1U
	mv '0Bz7lWLS0atxsbXZpT056ZnJnd1U' 'ds113_sub012.tgz'
	
process:
	tar -zxvf ds113_sub012.tgz
	
validate:
	python data.py

test:
	nosetests tests 


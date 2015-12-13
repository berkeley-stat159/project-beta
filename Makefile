.PHONY: all clean coverage

all: clean

data_process:
	wget -P ./date https://www.googledrive.com/host/0Bz7lWLS0atxsbXZpT056ZnJnd1U && mv '0Bz7lWLS0atxsbXZpT056ZnJnd1U' 'ds113_sub012.tgz' && tar -zxvf ds113_sub012.tgz

validate:
	cd /data && python data.py

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

coverage:
	nosetests code/utils data --with-coverage --cover-package=data  --cover-package=utils

test:
	cd code/utils/tests && nosetests *.py 
	cd data/tests && nosetests *.py

verbose:
	nosetests -v code/utils data

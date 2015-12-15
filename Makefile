.PHONY: all clean coverage

all: clean

data_download:
	wget -P ./data https://www.googledrive.com/host/0Bz7lWLS0atxsbXZpT056ZnJnd1U && cd data && mv '0Bz7lWLS0atxsbXZpT056ZnJnd1U' 'ds113_sub012.tgz' && tar -zxvf ds113_sub012.tgz
	wget -P ./data https://www.googledrive.com/host/0BxlqqubRo4V3WTVkSXVNTktuLW8 && cd data && mv '0BxlqqubRo4V3WTVkSXVNTktuLW8' 'smoothed_data.npy'

validate:
	cd data && python data.py

clean:
	find . -name "*.so" -o -name "*.pyc" -o -name "*.pyx.md5" | xargs rm -f

coverage:
	#nosetests code/utils data --with-coverage -cover-package=data  --cover-package=utils
	cd code/utils/tests && nosetests --with-coverage
	cd data/tests && nosetests --with-coverage

test:
	cd code/utils/tests && nosetests *.py 
	cd data/tests && nosetests *.py

preprocess_data:
	cd code && python dataprep_script.py
	cd code && python mask_generating.py 
	cd code && python filter_script.py
	cd code && python data_filtering.py
	cd code && python mask_generating.py

preprocess_description:
	cd code && python dataclean.py
	cd code && python gen_design_matrix.py

analysis:
	cd code && python description_modeling_ridge_regression.py
	cd code && python scenes_pred.py
	cd code && python nn.py

verbose:
	cd code/utils/tests && nosetests -v *.py
	cd data/tests && nosetests -v *.py 
	#nosetests -v code/utils data

paper_report:
	make clean -C paper
	make -C paper
test:
	cd utils/tests && nosetests *.py

coverage:
	cd utils/tests && nosetests *.py --with-coverage --cover-package=utils 

preprocess_data:
	python data_loading_script.py
	python dataprep_script.py
	python mask_generating.py 
	python filter_script.py
	python data_filtering.py
	python mask_generating.py

preprocess_description:
	python dataclean.py
	python gen_design_matrix.py

analysis:
	python description_modeling_ridge_regression.py
	python scenes_pred.py
	python nn.py
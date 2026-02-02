PYTHON_VERSION=3.12

create_venv:
	pyenv shell $(PYTHON_VERSION) &&\
	mkvirtualenv p1_env &&\
	pip install -r requirements.txt

run_pipeline:
	python -m src.pipeline
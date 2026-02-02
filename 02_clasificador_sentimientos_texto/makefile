PYTHON_VERSION=3.12

create_venv:
	pyenv shell $(PYTHON_VERSION) &&\
	mkvirtualenv p2_$(PYTHON_VERSION) &&\
	pip install --upgrade pip &&\
	pip install -r requirements.txt

run_pipeline:
	python -m src.pipeline
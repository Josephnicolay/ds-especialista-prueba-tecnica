PYTHON_VERSION=3.12

create_venv:
	pyenv shell $(PYTHON_VERSION) &&\
	mkvirtualenv p3_$(PYTHON_VERSION) &&\
	pip install --upgrade pip &&\
	pip install -r requirements.txt --index-url https://download.pytorch.org/whl/cu121

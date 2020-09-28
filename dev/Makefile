compile: model.stan
	python3 -c 'import pystan; pystan.StanModel("model.stan")'

freeze:
	rm -f requirements.txt
	. venv/bin/activate && pip freeze > requirements.txt


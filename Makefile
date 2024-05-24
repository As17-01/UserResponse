format:
	poetry run isort --sl src/ scripts/
	poetry run black src/ scripts/
	poetry run flake8 src/ scripts/ --ignore=D101,D102,D103,E402,N803,N806
	poetry run mypy 

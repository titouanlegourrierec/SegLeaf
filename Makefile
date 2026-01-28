.PHONY: lint format quality precommit clean help

SRC=.

lint:
	ruff check --fix $(SRC)

format:
	ruff format $(SRC)

quality: lint format

precommit:
	prek run --all-files

clean:
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.ipynb" -exec jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} +

help:
	@echo "\033[1;36mMakefile targets:\033[0m"
	@echo "  \033[1;32mlint\033[0m     : Check code with ruff."
	@echo "  \033[1;32mformat\033[0m   : Format code with ruff."
	@echo "  \033[1;32mquality\033[0m  : Run lint and format."
	@echo "  \033[1;32mclean\033[0m    : Remove temporary files."
	@echo "  \033[1;32mprecommit\033[0m: Run pre-commit hooks on all files."

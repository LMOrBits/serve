.PHONY: all

init:
	uv --version || pip install uv
	uv sync 

status:
	@uv run serve model --status

run:
	@uv run serve model --run

stop:
	@uv run serve model --stop

server-status:
	@uv run serve status



ruff-fix:
	@uv run ruff check --fix
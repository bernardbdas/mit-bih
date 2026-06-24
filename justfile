list-pythons:
    uv python list

init:
    echo "Creating data directories..."
    mkdir -p data/raw data/processed

    echo "Creating .venv (Python 3.12)..."
    uv venv .venv --python 3.12

    echo "Virtual environment created successfully!"

sync:
    uv sync --python .venv/bin/python

lock:
    uv lock

clean:
    rm -rf .venv
    echo "Deleted .venv/"

VENV_DIR=".venv"

if [ ! -d "$VENV_DIR" ]; then
  log_status "creating Python venv"
  uv venv "$VENV_DIR"
  uv pip install -e ".[dev]"
fi

source "$VENV_DIR/bin/activate"

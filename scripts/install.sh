#!/bin/bash
# Путь к скрипту: в zsh BASH_SOURCE не задан, используем ${(%):-%x}
if [ -n "${BASH_SOURCE[0]}" ]; then
  _SCRIPT_PATH="${BASH_SOURCE[0]}"
elif [ -n "$ZSH_VERSION" ]; then
  _SCRIPT_PATH="${(%):-%x}"
else
  _SCRIPT_PATH="$0"
fi
SCRIPT_DIR="$(cd "$(dirname "$_SCRIPT_PATH")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

set -e

# --- Функция-обертка для кроссплатформенного sed -i ---
sedi() {
    # $1: выражение sed (паттерн)
    # $2: файл
    if [ "$(uname -s)" = "Darwin" ]; then
        # macOS требует пустую строку '' после -i
        sed -i '' "$1" "$2"
    else
        # Linux не требует аргумента и не принимает пустую строку через пробел
        sed -i "$1" "$2"
    fi
}
# -----------------------------------------------------

if ! command -v uv >/dev/null 2>&1; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "Installing project dependencies..."
uv sync

echo "Installing chumpy (no build isolation: avoids 'No module named pip' in build env)..."
# chumpy собирается legacy-способом: ему нужны pip/setuptools в venv
uv pip install pip setuptools wheel
uv run python -m pip install --no-build-isolation "chumpy==0.70"

echo "Patching chumpy for Python >=3.11 (getargspec -> getfullargspec)..."
CH_FILE=$(uv run python -c "
import site, pathlib
p = next(p for p in site.getsitepackages() if 'site-packages' in p)
print(pathlib.Path(p) / 'chumpy' / 'ch.py')
")
if [ -n "$CH_FILE" ] && [ -f "$CH_FILE" ]; then
    # ИСПОЛЬЗУЕМ sedi ВМЕСТО sed -i
    sedi 's/inspect.getargspec/inspect.getfullargspec/g' "$CH_FILE"
    echo "Patched $CH_FILE"
else
    echo "chumpy/ch.py not found; skip patch"
fi

echo "Patching chumpy __init__.py for NumPy>=2.0 imports..."
CH_INIT=$(uv run python -c "
import site, pathlib
p = next(p for p in site.getsitepackages() if 'site-packages' in p)
print(pathlib.Path(p) / 'chumpy' / '__init__.py')
")
if [ -n "$CH_INIT" ] && [ -f "$CH_INIT" ]; then
    # ИСПОЛЬЗУЕМ sedi ВМЕСТО sed -i
    sedi 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/g' "$CH_INIT"
    echo "Patched $CH_INIT"
else
    echo "chumpy/__init__.py not found; skip patch"
fi

echo "Done. Run commands with: uv run python ..."

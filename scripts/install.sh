#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Проект требует Python 3.12. Если на системе по умолчанию 3.11 — задать явно перед poetry install:
#   poetry env use python3.12
# или
#   poetry env use /usr/bin/python3.12
if command -v poetry &>/dev/null; then
    POETRY_PY=$(poetry env info -p 2>/dev/null)
    if [ -n "$POETRY_PY" ] && [ -d "$POETRY_PY" ]; then
        ENV_VER=$("$POETRY_PY/bin/python" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
        if [ -n "$ENV_VER" ] && [ "$ENV_VER" != "3.12" ] && [ "$ENV_VER" != "3.13" ] && [ "$ENV_VER" != "3.14" ]; then
            echo "WARNING: Poetry venv uses Python $ENV_VER; project needs >=3.12. Run: poetry env use python3.12"
            echo "         Then: poetry env remove python && poetry install"
        fi
    fi
fi

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

echo "Installing poetry..."
pip install poetry

echo "Installing project dependencies..."
poetry install

echo "Installing chumpy (no build isolation: avoids 'No module named pip' in build env)..."
PIP_NO_BUILD_ISOLATION=1 poetry run pip install --no-build-isolation "chumpy==0.70"

echo "Patching chumpy for Python >=3.11 (getargspec -> getfullargspec)..."
CH_FILE=$(
poetry run python <<'PY'
import site
import pathlib

site_packages = next(p for p in site.getsitepackages() if "site-packages" in p)
print(pathlib.Path(site_packages) / "chumpy" / "ch.py")
PY
)
if [ -n "$CH_FILE" ] && [ -f "$CH_FILE" ]; then
    # ИСПОЛЬЗУЕМ sedi ВМЕСТО sed -i
    sedi 's/inspect.getargspec/inspect.getfullargspec/g' "$CH_FILE"
    echo "Patched $CH_FILE"
else
    echo "chumpy/ch.py not found; skip patch"
fi

echo "Patching chumpy __init__.py for NumPy>=2.0 imports..."
CH_INIT=$(
poetry run python <<'PY'
import site
import pathlib

site_packages = next(p for p in site.getsitepackages() if "site-packages" in p)
f = pathlib.Path(site_packages) / "chumpy" / "__init__.py"
print(f)
PY
)
if [ -n "$CH_INIT" ] && [ -f "$CH_INIT" ]; then
    # ИСПОЛЬЗУЕМ sedi ВМЕСТО sed -i
    sedi 's/from numpy import bool, int, float, complex, object, unicode, str, nan, inf/from numpy import nan, inf/g' "$CH_INIT"
    echo "Patched $CH_INIT"
else
    echo "chumpy/__init__.py not found; skip patch"
fi

echo "Done. To enter the venv run: poetry shell"
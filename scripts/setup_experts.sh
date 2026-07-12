#!/usr/bin/env bash
# Настраивает submodule-экспертов (Kinesis, Myohuman):
# инициализирует submodule и применяет локальные патчи.
# Универсальный инсталлятор: macOS (BSD sed) и Linux (GNU sed).
#
# Использование: ./scripts/setup_experts.sh [expert ...]
#   Без аргументов или "all" — настроить всех экспертов.
#   Иначе — только перечисленных: kinesis, myohuman.
#   Пример: ./scripts/setup_experts.sh kinesis myohuman

set -e

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

AVAILABLE_EXPERTS="kinesis myohuman"

# Разбор аргументов: по умолчанию все
if [ $# -eq 0 ] || [ "$*" = "all" ]; then
    EXPERTS="$AVAILABLE_EXPERTS"
else
    EXPERTS=""
    for arg in "$@"; do
        case " $AVAILABLE_EXPERTS " in
            *" $arg "*) EXPERTS="$EXPERTS $arg" ;;
            *) echo "Unknown expert: $arg (available: $AVAILABLE_EXPERTS)" >&2; exit 1 ;;
        esac
    done
fi
EXPERTS=$(echo $EXPERTS)
echo "Setting up expert(s): $EXPERTS"

# --- Платформа: Darwin = macOS (BSD sed), иначе Linux (GNU sed) ---
SED_IS_BSD=
case "$(uname -s)" in
    Darwin)  SED_IS_BSD=1 ;;
    Linux)   ;;
    *)       echo "Warning: unknown OS $(uname -s), assuming GNU sed"
esac

# --- Функция-обертка для кроссплатформенного sed -i ---
# macOS/BSD: sed -i '' "expr" file
# Linux:    sed -i "expr" file
sedi() {
    if [ -n "$SED_IS_BSD" ]; then
        sed -i '' "$1" "$2"
    else
        sed -i "$1" "$2"
    fi
}
# Несколько выражений подряд: run_sedi_multi -e 's/.../g' -e 's/.../g' -- "$file"
run_sedi_multi() {
    local args=() file
    while [ $# -gt 0 ]; do
        if [ "$1" = "--" ]; then
            shift
            file="$1"
            break
        fi
        args+=("$1")
        shift
    done
    if [ -z "$file" ] || [ ${#args[@]} -eq 0 ]; then return 1; fi
    if [ -n "$SED_IS_BSD" ]; then
        sed -i '' "${args[@]}" "$file"
    else
        sed -i "${args[@]}" "$file"
    fi
}
# -----------------------------------------------------

echo "Platform: $(uname -s)"
echo "Initializing submodules..."
for ex in $EXPERTS; do
    case "$ex" in
        kinesis)            path="src/myotrainer/experts/Kinesis" ;;
        myohuman)           path="src/myotrainer/experts/Myohuman" ;;
        *)                  path="" ;;
    esac
    [ -n "$path" ] && git submodule update --init --recursive "$path" && echo "  Inited $ex"
done
echo "Pulling LFS files in selected submodules..."
for ex in $EXPERTS; do
    case "$ex" in
        kinesis)            path="src/myotrainer/experts/Kinesis" ;;
        myohuman)           path="src/myotrainer/experts/Myohuman" ;;
        *)                  path="" ;;
    esac
    if [ -n "$path" ] && [ -d "$REPO_ROOT/$path/.git" ]; then
        (cd "$REPO_ROOT/$path" && git lfs pull) && echo "  LFS pulled $ex"
    fi
done

# ========== Kinesis ==========
if case " $EXPERTS " in *" kinesis "*) true;; *) false;; esac; then
    KINESIS_ROOT="$REPO_ROOT/src/myotrainer/experts/Kinesis"
    if [ ! -d "$KINESIS_ROOT" ]; then
        echo "Kinesis submodule not found at $KINESIS_ROOT; skip Kinesis patches"
    else
        echo "Patching Kinesis..."

        # visual_capsule.py: mjv_makeConnector + распакованные координаты -> mjv_connector + point1, point2
        VC="$KINESIS_ROOT/src/utils/visual_capsule.py"
        if [ -f "$VC" ]; then
            sedi 's/mjv_makeConnector/mjv_connector/g' "$VC"
            sedi 's/point1\[0\], point1\[1\], point1\[2\],/point1,/g' "$VC"
            sedi 's/point2\[0\], point2\[1\], point2\[2\]/point2/g' "$VC"
            echo "  Patched $VC"
        else
            echo "  $VC not found; skip"
        fi

        # agent_humanoid.py: torch.load — добавить weights_only=False (совместимость с PyTorch 2+ и старыми чекпоинтами)
        AH="$KINESIS_ROOT/src/agents/agent_humanoid.py"
        if [ -f "$AH" ]; then
            sedi 's/map_location=self\.device)/map_location=self.device, weights_only=False)/g' "$AH"
            echo "  Patched $AH"
        else
            echo "  $AH not found; skip"
        fi

        KINESIS_DATA_XML="$KINESIS_ROOT/data/xml"
        if [ -d "$KINESIS_DATA_XML" ]; then
            SED_JOINT_R=(
                -e 's/knee_angle_r_translation2_constraint/knee_angle_translation2_constraint_r/g'
                -e 's/knee_angle_r_translation1_constraint/knee_angle_translation1_constraint_r/g'
                -e 's/knee_angle_r_rotation2_constraint/knee_angle_rotation2_constraint_r/g'
                -e 's/knee_angle_r_rotation3_constraint/knee_angle_rotation3_constraint_r/g'
                -e 's/knee_angle_r_beta_translation2_constraint/knee_angle_beta_translation2_constraint_r/g'
                -e 's/knee_angle_r_beta_translation1_constraint/knee_angle_beta_translation1_constraint_r/g'
                -e 's/knee_angle_r_beta_rotation1_constraint/knee_angle_beta_rotation1_constraint_r/g'
                -e 's/knee_angle_l_translation2_constraint/knee_angle_translation2_constraint_l/g'
                -e 's/knee_angle_l_translation1_constraint/knee_angle_translation1_constraint_l/g'
                -e 's/knee_angle_l_rotation2_constraint/knee_angle_rotation2_constraint_l/g'
                -e 's/knee_angle_l_rotation3_constraint/knee_angle_rotation3_constraint_l/g'
                -e 's/knee_angle_l_beta_translation2_constraint/knee_angle_beta_translation2_constraint_l/g'
                -e 's/knee_angle_l_beta_translation1_constraint/knee_angle_beta_translation1_constraint_l/g'
                -e 's/knee_angle_l_beta_rotation1_constraint/knee_angle_beta_rotation1_constraint_l/g'
                -e 's/knee_angle_r_translation2/knee_angle_translation2_r/g'
                -e 's/knee_angle_r_translation1/knee_angle_translation1_r/g'
                -e 's/knee_angle_r_rotation2/knee_angle_rotation2_r/g'
                -e 's/knee_angle_r_rotation3/knee_angle_rotation3_r/g'
                -e 's/knee_angle_r_beta_translation2/knee_angle_beta_translation2_r/g'
                -e 's/knee_angle_r_beta_translation1/knee_angle_beta_translation1_r/g'
                -e 's/knee_angle_r_beta_rotation1/knee_angle_beta_rotation1_r/g'
                -e 's/knee_angle_l_translation2/knee_angle_translation2_l/g'
                -e 's/knee_angle_l_translation1/knee_angle_translation1_l/g'
                -e 's/knee_angle_l_rotation2/knee_angle_rotation2_l/g'
                -e 's/knee_angle_l_rotation3/knee_angle_rotation3_l/g'
                -e 's/knee_angle_l_beta_translation2/knee_angle_beta_translation2_l/g'
                -e 's/knee_angle_l_beta_translation1/knee_angle_beta_translation1_l/g'
                -e 's/knee_angle_l_beta_rotation1/knee_angle_beta_rotation1_l/g'
            )
            # Рекурсивно находим все XML с knee_angle_r_ или knee_angle_l_ и патчим
            KNEE_FILES=$(grep -rl "knee_angle_r_\|knee_angle_l_" "$KINESIS_DATA_XML" --include="*.xml" 2>/dev/null)
            for f in $KNEE_FILES; do
                run_sedi_multi "${SED_JOINT_R[@]}" -- "$f"
                echo "  Patched $(basename "$f") ($(dirname "$f" | sed "s|$KINESIS_DATA_XML/||"))"
            done
        fi

        # Загрузка ассетов и моделей Kinesis 2.0
        KINESIS_DATA="$KINESIS_ROOT/data"
        ASSETS="$REPO_ROOT/downloads/Kinesis_assets"

        # Определяем python из uv venv
        VENV_PY="$REPO_ROOT/.venv/bin/python"
        if [ ! -x "$VENV_PY" ]; then
            VENV_PY="uv run --project $REPO_ROOT python"
        fi

        # Ассеты и модели: предпочитаем локальную копию (downloads/Kinesis_assets),
        # иначе скачиваем через download_assets.py + download_models.py
        if [ -d "$ASSETS" ]; then
            echo "Copying Kinesis 2.0 assets from downloads/Kinesis_assets..."
            # SMPL
            mkdir -p "$KINESIS_DATA/smpl"
            [ -f "$ASSETS/smpl/SMPL_NEUTRAL.pkl" ] && cp "$ASSETS/smpl/SMPL_NEUTRAL.pkl" "$KINESIS_DATA/smpl/" && echo "  smpl/SMPL_NEUTRAL.pkl"
            # Motion dicts
            [ -f "$ASSETS/kit_test_motion_dict.pkl" ] && cp "$ASSETS/kit_test_motion_dict.pkl" "$KINESIS_DATA/" && echo "  kit_test_motion_dict.pkl"
            [ -f "$ASSETS/kit_train_motion_dict.pkl" ] && cp "$ASSETS/kit_train_motion_dict.pkl" "$KINESIS_DATA/" && echo "  kit_train_motion_dict.pkl"
            # Initial poses (per-model subdirs: legs, legs_abs, legs_back, fullbody)
            if [ -d "$ASSETS/initial_pose" ]; then
                cp -R "$ASSETS/initial_pose" "$KINESIS_DATA/"
                echo "  initial_pose/ ($(ls "$ASSETS/initial_pose" | tr '\n' ' '))"
            fi
            # Trained models (per-model subdirs: legs, legs_abs, legs_back)
            if [ -d "$ASSETS/trained_models" ]; then
                cp -R "$ASSETS/trained_models" "$KINESIS_DATA/"
                echo "  trained_models/ ($(ls "$ASSETS/trained_models" | grep -v README | tr '\n' ' '))"
            fi
            echo "  Kinesis assets copied."
        else
            echo "Downloading Kinesis 2.0 assets from Hugging Face..."
            (cd "$KINESIS_ROOT" && ($VENV_PY -c "import huggingface_hub" 2>/dev/null || $VENV_PY -m pip install -q huggingface_hub) && $VENV_PY src/utils/download_assets.py --branch kinesis-2.0) || echo "  Warning: assets download failed. You can run manually: cd $KINESIS_ROOT && python src/utils/download_assets.py --branch kinesis-2.0"
            echo "Downloading Kinesis 2.0 models from Hugging Face..."
            (cd "$KINESIS_ROOT" && $VENV_PY src/utils/download_models.py) || echo "  Warning: model download failed. You can run manually: cd $KINESIS_ROOT && python src/utils/download_models.py"
        fi
    fi
fi

echo "Done. Experts are ready."

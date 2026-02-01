#!/usr/bin/env bash
# Настраивает submodule-экспертов (Kinesis, myochallenge-lattice):
# инициализирует submodule и применяет локальные патчи.
# Универсальный инсталлятор: macOS (BSD sed) и Linux (GNU sed).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

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
git submodule update --init --recursive

# ========== Kinesis ==========
KINESIS_ROOT="$REPO_ROOT/src/arnold/experts/Kinesis"
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

    # data/xml: патчи именования joint (knee_angle_X_r -> knee_angle_X_r, суффикс _r/_l в конец)
    KINESIS_DATA_XML="$KINESIS_ROOT/data/xml"
    if [ -d "$KINESIS_DATA_XML" ]; then
        # myolegs_assets.xml и myolegs.xml — одни и те же замены (сначала _constraint, потом короткие имена)
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
        for f in myolegs_assets.xml myolegs.xml; do
            if [ -f "$KINESIS_DATA_XML/$f" ]; then
                run_sedi_multi "${SED_JOINT_R[@]}" -- "$KINESIS_DATA_XML/$f"
                echo "  Patched $f"
            fi
        done
        # smpl_humanoid.xml: убрать пустую строку после первой, поправить density
        SMPL="$KINESIS_DATA_XML/smpl_humanoid.xml"
        if [ -f "$SMPL" ]; then
            run_sedi_multi \
                -e '2{/^$/d;}' \
                -e 's/density="449\.071517"/density="449.071446"/g' \
                -e 's/density="423\.206882"/density="423.206917"/g' \
                -e 's/density="434\.701472"/density="434.701429"/g' \
                -e 's/density="407\.383947"/density="407.383984"/g' \
                -e 's/density="400\.55248"/density="400.552479"/g' \
                -e 's/density="403\.679838"/density="403.679819"/g' \
                -- "$SMPL"
            echo "  Patched smpl_humanoid.xml"
        fi
    fi

    # Копирование ассетов из downloads/Kinesis_assets (LFS) в data эксперта
    ASSETS="$REPO_ROOT/downloads/Kinesis_assets"
    KINESIS_DATA="$KINESIS_ROOT/data"
    if [ -d "$ASSETS" ]; then
        echo "Copying Kinesis assets from downloads/Kinesis_assets..."
        mkdir -p "$KINESIS_DATA/smpl" "$KINESIS_DATA/initial_pose"
        [ -f "$ASSETS/smpl/SMPL_NEUTRAL.pkl" ] && cp "$ASSETS/smpl/SMPL_NEUTRAL.pkl" "$KINESIS_DATA/smpl/" && echo "  smpl/SMPL_NEUTRAL.pkl"
        [ -f "$ASSETS/initial_pose/initial_pose_test.pkl" ] && cp "$ASSETS/initial_pose/initial_pose_test.pkl" "$KINESIS_DATA/initial_pose/" && echo "  initial_pose/initial_pose_test.pkl"
        [ -f "$ASSETS/initial_pose/initial_pose_train.pkl" ] && cp "$ASSETS/initial_pose/initial_pose_train.pkl" "$KINESIS_DATA/initial_pose/" && echo "  initial_pose/initial_pose_train.pkl"
        [ -f "$ASSETS/kit_test_motion_dict.pkl" ] && cp "$ASSETS/kit_test_motion_dict.pkl" "$KINESIS_DATA/" && echo "  kit_test_motion_dict.pkl"
        [ -f "$ASSETS/kit_train_motion_dict.pkl" ] && cp "$ASSETS/kit_train_motion_dict.pkl" "$KINESIS_DATA/" && echo "  kit_train_motion_dict.pkl"
        echo "  Kinesis assets copied."
        # Загрузка модели эксперта с Hugging Face
        if ! [ -f "$KINESIS_DATA/trained_models/kinesis-moe-imitation/model.pth" ]; then
            echo "Downloading Kinesis model from Hugging Face..."
            poetry -C "$REPO_ROOT" env use python3.12 2>/dev/null || true
            VENV_PY="$(poetry -C "$REPO_ROOT" env info -p 2>/dev/null)/bin/python"
            if [ -x "$VENV_PY" ]; then
                (cd "$KINESIS_ROOT" && ("$VENV_PY" -c "import huggingface_hub" 2>/dev/null || "$VENV_PY" -m pip install -q huggingface_hub) && "$VENV_PY" src/utils/download_model.py --repo_id amathislab/kinesis-moe-imitation) || echo "  Warning: model download failed (run from arnold root after: poetry install)."
            else
                (cd "$KINESIS_ROOT" && (poetry -C "$REPO_ROOT" run python -c "import huggingface_hub" 2>/dev/null || poetry -C "$REPO_ROOT" run pip install -q huggingface_hub) && poetry -C "$REPO_ROOT" run python src/utils/download_model.py --repo_id amathislab/kinesis-moe-imitation) || echo "  Warning: model download failed (run from arnold root after: poetry install)."
            fi
        else
            echo "Kinesis model already present."
        fi
    else
        echo "downloads/Kinesis_assets not found; run git lfs pull if needed, then re-run setup."
    fi
fi

echo "Done. Experts are ready."

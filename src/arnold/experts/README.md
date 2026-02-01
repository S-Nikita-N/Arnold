# Эксперты для Arnold

Эта директория содержит репозитории с предобученными экспертами.

## Репозитории

### Kinesis (MyoLegs - Walk to point)
- URL: https://github.com/amathislab/Kinesis
- Назначение: Эксперт для задачи Walk to point на MyoLegs (80 мышц)
- Статус: Подтвержден, работает

### myochallenge-lattice (MyoArm - Object relocate)
- URL: https://github.com/amathislab/myochallenge-lattice
- Назначение: Эксперт для задачи Object relocate на MyoArm (48 мышц без кисти)
- Статус: Требует проверки

---

## Установка

### Общие шаги

#### 1. Клонирование репозиториев как Git Submodules

Выполните следующие команды в этой директории:

```bash
# Клонировать с submodules
git clone --recurse-submodules <repository-url>

# Или если уже склонировали без submodules:
git submodule update --init --recursive
```

---

### Установка Kinesis (MyoLegs эксперт)

В репозитории Arnold всё делается одним скриптом из корня репо:

```bash
./scripts/setup_experts.sh
```

Скрипт:
1. Инициализирует submodule Kinesis
2. Применяет патчи к коду и XML
3. Копирует ассеты из `downloads/Kinesis_assets/` (Git LFS) в `Kinesis/data/`:
   - `smpl/SMPL_NEUTRAL.pkl`
   - `kit_train_motion_dict.pkl`, `kit_test_motion_dict.pkl`
   - `initial_pose/initial_pose_train.pkl`, `initial_pose/initial_pose_test.pkl`
4. Скачивает модель эксперта с Hugging Face: `amathislab/kinesis-moe-imitation`

После клона репо выполните `git lfs pull` (если используете LFS), затем `./scripts/setup_experts.sh`.

Дополнительная модель для target goal reaching (по желанию):
```bash
cd src/arnold/experts/Kinesis
python src/utils/download_model.py --repo_id amathislab/kinesis-target-goal-reach
```

---

#### Ручные шаги (только если не используете setup_experts.sh)

#### Шаг 1: Фикс кода под используемые версии библиотек

```bash
sed -i '' "s/torch.load(checkpoint_path, map_location=self.device)/torch.load(checkpoint_path, map_location=self.device, weights_only=False)/" src/agents/agent_humanoid.py

sed -i '' \  
   -e 's/mjv_makeConnector/mjv_connector/' \
   -e 's/point1\[0], point1\[1], point1\[2]/point1/' \
   -e 's/point2\[0], point2\[1], point2\[2]/point2/' \
   src/utils/visual_capsule.py
```

#### Шаг 2: Фикс именования суставов колена в MyoLegs модели

Суставы левого колена в MyoLegs имеют неконсистентный нейминг (`knee_angle_l_XXX` вместо `knee_angle_XXX_l`).
Исправляем для единообразного парсинга:

```bash
# Исправляем суффиксы для обеих сторон (_l_ и _r_ -> _XXX_l и _XXX_r)
sed -i '' \
   -e 's/knee_angle_l_\([a-z_0-9]*\)/knee_angle_\1_l/g' \
   -e 's/knee_angle_r_\([a-z_0-9]*\)/knee_angle_\1_r/g' \
   data/xml/myolegs_assets.xml data/xml/myolegs.xml
```

Это исправит:
- `knee_angle_l_translation2` → `knee_angle_translation2_l`
- `knee_angle_l_rotation2` → `knee_angle_rotation2_l`
- `knee_angle_l_beta_translation1` → `knee_angle_beta_translation1_l`
- и т.д.

---

### Установка myochallenge-lattice (MyoArm эксперт)

**Статус**: Требует проверки и уточнения плана установки.

После проверки репозитория здесь будет добавлен полный план установки.

---

## Структура

После клонирования и установки структура будет следующей:

```
experts/
├── README.md
├── __init__.py
├── expert_wrapper.py
├── kinesis_expert.py          # (будет создан)
├── myochallenge_expert.py     # (будет создан)
├── Kinesis/
│   ├── data/
│   │   ├── smpl/
│   │   │   └── SMPL_NEUTRAL.pkl
│   │   └── trained_models/
│   │       ├── kinesis-moe-imitation/
│   │       │   └── model.pth
│   │       └── kinesis-target-goal-reach/
│   │           └── model.pth
│   └── ...
└── myochallenge-lattice/
    └── ...
```

---

## Использование

Эксперты будут использоваться через wrapper классы в `fullbody/src/myohuman/arnold/experts/`:

- `expert_wrapper.py` - базовый класс для всех экспертов
- `kinesis_wrapper.py` - wrapper для Kinesis (MyoLegs)
- `myochallenge_wrapper.py` - wrapper для myochallenge-lattice (MyoArm)

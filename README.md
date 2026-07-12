# MyoTrainer

Обучаемый агент на базе On-Policy Behavior Cloning (OBC) и эксперта Kinesis (MyoLegs).

> **За основу взята работа Arnold** (реализация и адаптация):
> Chiappa A. S., An B., Simos M., Li C., Mathis A.
> *Arnold: A Generalist Muscle Transformer Policy.*
> 2025. [arXiv:2508.18066](https://arxiv.org/abs/2508.18066).
>
> MyoTrainer — переработанная имплементация Arnold под задачу управления
> полнотелой моделью MyoHuman (338 мышечных актуаторов, имитация motion
> capture траекторий из AMASS) с расширениями: иерархический декодер
> действий, грануляции наблюдений и мышечных групп, дистилляция
> Kinesis MoE в трансформер-политику, многостадийный пайплайн обучения.

---

## Клонирование на новом компьютере

Нужны: **Git**, **Git LFS**, **uv** (поставит Python 3.12+ сам; `install.sh` установит uv при отсутствии).

### 1. Установить Git LFS (один раз на системе)

```bash
# macOS (Homebrew)
brew install git-lfs
git lfs install

# Linux (apt)
sudo apt install git-lfs
git lfs install
```

### 2. Клонировать репозиторий с submodules и LFS

```bash
git clone --recurse-submodules <URL-репозитория> MyoTrainer
cd MyoTrainer
git lfs pull
```

- `--recurse-submodules` — подтягивает Kinesis, myochallenge-lattice и Myohuman.
- `git lfs pull` — подтягивает тяжёлые файлы из `downloads/Kinesis_assets/` (SMPL, motion dicts, initial poses). LFS-файлы внутри субмодулей (в т.ч. Myohuman) подтягиваются скриптом `setup_experts.sh`.

Если репо уже склонирован без submodules:

```bash
git submodule update --init --recursive
git lfs pull
```

### 3. Установить зависимости и окружение

Из корня репозитория:

```bash
./scripts/install.sh
```

Окружение создаётся в `.venv`; команды запускаются через `uv run python ...` (или `source .venv/bin/activate`).

### 4. Настроить экспертов (патчи + копирование ассетов + загрузка модели)

```bash
./scripts/setup_experts.sh
```

Скрипт:
- инициализирует submodules (если ещё не сделано);
- подтягивает LFS-файлы во всех субмодулях (в т.ч. Myohuman);
- применяет патчи к коду и XML в Kinesis;
- копирует ассеты из `downloads/Kinesis_assets/` в `src/arnold/experts/Kinesis/data/`;
- скачивает модель эксперта с Hugging Face (`kinesis-moe-imitation`).

После этого можно запускать обучение и оценку (см. конфиги в `cfg/`).

---

## Краткий чеклист

| Шаг | Команда |
|-----|---------|
| 1 | `git lfs install` (один раз) |
| 2 | `git clone --recurse-submodules <url> MyoTrainer && cd MyoTrainer` |
| 3 | `git lfs pull` |
| 4 | `./scripts/install.sh` |
| 5 | `./scripts/setup_experts.sh` |

---

## Структура

- `cfg/` — конфиги Hydra (env, learning, run).
- `src/arnold/` — код MyoTrainer (policy, trainer, experts).
- `downloads/Kinesis_assets/` — ассеты для Kinesis (Git LFS).
- `scripts/install.sh` — установка uv и зависимостей.
- `scripts/setup_experts.sh` — патчи и подготовка данных экспертов.

Подробнее по экспертам: [src/arnold/experts/README.md](src/arnold/experts/README.md).

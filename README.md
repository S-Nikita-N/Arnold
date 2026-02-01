# Arnold

Обучаемый агент на базе On-Policy Behavior Cloning (OBC) и эксперта Kinesis (MyoLegs).

---

## Клонирование на новом компьютере

Нужны: **Git**, **Git LFS**, **Python 3.12+**, **Poetry**.

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
git clone --recurse-submodules <URL-репозитория> arnold
cd arnold
git lfs pull
```

- `--recurse-submodules` — подтягивает Kinesis и myochallenge-lattice.
- `git lfs pull` — подтягивает тяжёлые файлы из `downloads/Kinesis_assets/` (SMPL, motion dicts, initial poses).

Если репо уже склонирован без submodules:

```bash
git submodule update --init --recursive
git lfs pull
```

### 3. Установить зависимости и окружение

Из корня репозитория:

```bash
./scripts/install.sh
poetry shell
```

### 4. Настроить экспертов (патчи + копирование ассетов + загрузка модели)

```bash
./scripts/setup_experts.sh
```

Скрипт:
- инициализирует submodules (если ещё не сделано);
- применяет патчи к коду и XML в Kinesis;
- копирует ассеты из `downloads/Kinesis_assets/` в `src/arnold/experts/Kinesis/data/`;
- скачивает модель эксперта с Hugging Face (`kinesis-moe-imitation`).

После этого можно запускать обучение и оценку (см. конфиги в `cfg/`).

---

## Краткий чеклист

| Шаг | Команда |
|-----|---------|
| 1 | `git lfs install` (один раз) |
| 2 | `git clone --recurse-submodules <url> arnold && cd arnold` |
| 3 | `git lfs pull` |
| 4 | `./scripts/install.sh` → `poetry shell` |
| 5 | `./scripts/setup_experts.sh` |

---

## Структура

- `cfg/` — конфиги Hydra (env, learning, run).
- `src/arnold/` — код Arnold (policy, trainer, experts).
- `downloads/Kinesis_assets/` — ассеты для Kinesis (Git LFS).
- `scripts/install.sh` — установка Poetry и зависимостей.
- `scripts/setup_experts.sh` — патчи и подготовка данных экспертов.

Подробнее по экспертам: [src/arnold/experts/README.md](src/arnold/experts/README.md).

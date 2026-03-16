# Action Granulation для Arnold

Модуль грануляции мышц и sparse attention для action decoder.

## Компоненты

### 1. Muscle Parser (`muscle_parser.py`)
Парсит MuJoCo XML и извлекает список actuator names в порядке модели.
- `parse_muscles_myohuman()` — для MyoHuman по умолчанию
- `parse_muscles_from_xml(path)` — для кастомного XML

### 2. Muscle Granulator (`muscle_granulator.py`)
Две стратегии группировки:

**Стратегия 1 (anatomical):** Анатомическая агрегация фасций
- PECM1/2/3 → pectoralis
- glmax1/2/3 → gluteus_max
- IL_*, LTpT_*, MF_*, QL_* → spine groups
- ~60 групп для MyoHuman

**Стратегия 2 (functional):** Функциональные синергии
- По действию на сустав: hip_flexors, knee_extensors, ankle_plantarflexors...
- ~34 группы для MyoHuman

### 3. Attention Mask Config (`attention_mask_config.py`)
Гибкая настройка масок для каждого decoder block:
- `full` — полное внимание
- `block_diagonal` — только внутри групп мышц
- `hierarchical` — group tokens + muscles (требует доп. токенов)
- `sparse_all`, `progressive`, `alternating` — стратегии по слоям

### 4. Sparse Action Decoder (`sparse_action_decoder.py`)
Decoder с per-layer mask support. Использует `attn_mask` в MultiheadAttention.

## Использование

### В конфиге (cfg/learning/arnold.yaml)
```yaml
transformer:
  # ... остальные параметры ...
  action_granulation: anatomical  # или functional
  action_decoder_mask_strategy: sparse_all  # или progressive
```

### Программно
```python
from arnold.action import parse_muscles_myohuman, granulate, build_attention_mask

muscles = parse_muscles_myohuman()
grouping = granulate(muscles, strategy="anatomical")
# grouping.n_groups, grouping.muscle_to_group, grouping.groups
```

## Сложность

- **Было:** O(A²) self-attention по A=338 мышцам
- **Стало (block_diagonal):** O(Σ A_g²) где A_g — размер группы
- Для anatomical: ~60 групп, макс. группа ~80 (spine_extensors) → значительный выигрыш

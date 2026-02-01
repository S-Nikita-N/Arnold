# Kinesis assets

Файлы для эксперта Kinesis хранятся здесь и подтягиваются через Git LFS при `git clone` / `git pull`.

Ожидаемая структура (копируется в `src/arnold/experts/Kinesis/data/` скриптом `scripts/setup_experts.sh`):

```
Kinesis_assets/
├── smpl/
│   └── SMPL_NEUTRAL.pkl
├── initial_pose/
│   ├── initial_pose_test.pkl
│   └── initial_pose_train.pkl
├── kit_test_motion_dict.pkl
└── kit_train_motion_dict.pkl
```

Модель эксперта (kinesis-moe-imitation) скачивается с Hugging Face при запуске `setup_experts.sh`.

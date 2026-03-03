"""
SamplingProfiler — покомпонентный профайлер горячего пути семплирования.

Аккумулирует суммарное время и количество вызовов по ключам.
Используется только в main process (pid==0), без overhead в воркерах.

Использование:
    profiler = SamplingProfiler()

    profiler.tick("env_step")
    env.step(action)
    profiler.tock("env_step")

    # В конце эпохи:
    print(profiler.report())
    profiler.reset()
"""

import time
from typing import Dict, Optional


class SamplingProfiler:
    def __init__(self):
        self._totals: Dict[str, float] = {}  # key → суммарное время (сек)
        self._counts: Dict[str, int] = {}    # key → кол-во вызовов
        self._starts: Dict[str, float] = {}  # key → время последнего tick

    def tick(self, key: str) -> None:
        self._starts[key] = time.perf_counter()

    def tock(self, key: str) -> None:
        elapsed = time.perf_counter() - self._starts[key]
        self._totals[key] = self._totals.get(key, 0.0) + elapsed
        self._counts[key] = self._counts.get(key, 0) + 1

    def reset(self) -> None:
        self._totals.clear()
        self._counts.clear()
        self._starts.clear()

    def to_dict(self) -> Dict[str, any]:
        return {"totals": dict(self._totals), "counts": dict(self._counts)}

    @classmethod
    def from_dict(cls, data: Dict[str, any]) -> "SamplingProfiler":
        p = cls()
        p._totals = data["totals"]
        p._counts = data["counts"]
        return p

    def total_tracked(self) -> float:
        """Суммарное время по всем ключам верхнего уровня (без policy_*)."""
        return sum(
            v for k, v in self._totals.items()
            if not k.startswith("_")
        )

    def report(self, total_steps: Optional[int] = None) -> str:
        """
        Возвращает текстовый отчёт.

        Args:
            total_steps: если передать — считает среднее на шаг.
        """
        if not self._totals:
            return "  [profiler] no data"

        # Считаем суммарное время всех top-level ключей для процентов
        top_keys = [k for k in self._totals if not k.startswith("  ")]
        total_wall = sum(self._totals[k] for k in top_keys)

        lines = []
        # Порядок вывода
        ordered_keys = [
            "parser.get_obs",
            "policy.forward",
            "  normalizer",
            "  body_tokenizer",
            "  vocab_embed_obs",
            "  transformer_encoder",
            "  vocab_embed_act",
            "  action_decoder",
            "  value_decoder",
            "expert.get_action",
            "env.step",
            "parser.update",
            "memory.append",
        ]
        # добавляем ключи которые есть, но не в ordered_keys
        extra = [k for k in self._totals if k not in ordered_keys]

        for key in ordered_keys + extra:
            if key not in self._totals:
                continue
            t = self._totals[key]
            n = self._counts[key]
            avg_ms = t / n * 1000
            indent = "    " if key.startswith("  ") else "  "
            label = key.strip()

            if key.startswith("  "):
                # sub-компонент policy — считаем % от policy.forward
                policy_t = self._totals.get("policy.forward", t)
                pct = t / policy_t * 100 if policy_t > 0 else 0
                lines.append(
                    f"{indent}{label:<28} {avg_ms:7.3f} ms/step  "
                    f"({pct:5.1f}% of policy.forward)"
                )
            else:
                pct = t / total_wall * 100 if total_wall > 0 else 0
                lines.append(
                    f"{indent}{label:<28} {avg_ms:7.3f} ms/step  "
                    f"({pct:5.1f}%)"
                )

        lines.append(f"  {'─' * 55}")
        total_ms = total_wall / self._counts.get("env.step", 1) * 1000
        lines.append(f"  {'TOTAL (tracked)':<28} {total_ms:7.3f} ms/step")

        return "\n".join(lines)

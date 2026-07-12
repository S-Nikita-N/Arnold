"""
Profiler — единый профайлер времени и GPU-памяти с вложенными секциями.

Два независимых режима (можно включать/выключать по отдельности):
  time_enabled  — замеряет wall-clock time (perf_counter)
  mem_enabled   — замеряет GPU memory (cuda allocated / peak)

Использование:
    profiler = Profiler(device=dev)  # dev, e.g. cuda:0 / mps / cpu

    # Включить время (sampling worker)
    profiler.time_enabled = True

    # Включить память (update_params, first mini-batch)
    profiler.mem_enabled = True

    with profiler.section("forward"):
        with profiler.section("trunk"):
            ...
        with profiler.section("experts"):
            ...

    print(profiler.report())
    profiler.reset()

Когда оба флага False — section() просто yield, без overhead.
"""

import time
import torch
import contextlib

from typing import Any


########################################
#               Profiler               #
########################################


class Profiler:
    """
    Единый профайлер с вложенными секциями и деревянным отчётом.

    Записывает время и/или GPU-память для каждой секции.
    Вложенность отслеживается через стек — секции внутри других секций
    отображаются с отступами в стиле tree(1).
    """

    def __init__(self, device=None):
        self.device = device
        self.time_enabled: bool = False
        self.mem_enabled: bool = False

        # Shared nesting stack
        self._stack: list[str] = []

        # Records: {'depth', 'name', 'meta', 'time', 'mem_before',
        #  'mem_after', 'mem_delta', 'mem_peak'}
        self._records: list[dict[str, Any]] = []

    ########################################
    #           Internal helpers           #
    ########################################

    def _sync_mem(self) -> tuple:
        """(cur_mb, peak_mb) — текущее и пиковое выделение GPU."""
        if not torch.cuda.is_available():
            return 0.0, 0.0
        torch.cuda.synchronize(self.device)
        cur = torch.cuda.memory_allocated(self.device) / 1024**2
        peak = torch.cuda.max_memory_allocated(self.device) / 1024**2
        return cur, peak

    ########################################
    #              Public API              #
    ########################################

    @contextlib.contextmanager
    def section(self, name: str, meta: str = ""):
        if not self.time_enabled and not self.mem_enabled:
            yield
            return

        depth = len(self._stack)
        self._stack.append(name)

        rec: dict[str, Any] = {
            "depth": depth,
            "name": name,
            "meta": meta,
            "time": None,
            "mem_before": None,
            "mem_after": None,
            "mem_delta": None,
            "mem_peak": None,
        }
        self._records.append(rec)

        t0 = time.perf_counter() if self.time_enabled else None
        mem_before = self._sync_mem()[0] if self.mem_enabled else None
        if mem_before is not None:
            rec["mem_before"] = mem_before

        try:
            yield
        finally:
            if t0 is not None:
                rec["time"] = time.perf_counter() - t0
            if self.mem_enabled:
                after_cur, peak = self._sync_mem()
                rec["mem_after"] = after_cur
                rec["mem_delta"] = after_cur - rec["mem_before"]
                rec["mem_peak"] = peak
            self._stack.pop()

    def reset(self) -> None:
        self._records.clear()
        self._stack.clear()
        if self.mem_enabled and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

    ########################################
    #            Serialization             #
    ########################################

    def to_dict(self) -> dict[str, Any]:
        return {"records": list(self._records)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Profiler":
        p = cls()
        p._records = data["records"]
        return p

    ########################################
    #         Tree layout helpers          #
    ########################################

    def _build_tree_layout(self, records: list[dict[str, Any]]):
        """Вычисляет is_last и prefixes для tree-отображения."""
        n = len(records)

        # is_last[i]: запись i — последний ребёнок своего родителя
        is_last = [True] * n
        for i in range(n - 1):
            d = records[i]["depth"]
            for j in range(i + 1, n):
                dj = records[j]["depth"]
                if dj < d:
                    break
                if dj == d:
                    is_last[i] = False
                    break

        # Build tree prefixes (depth 0 тоже получает ├─ / └─)
        prefixes: list[str] = []
        for i, rec in enumerate(records):
            depth = rec["depth"]
            parts: list[str] = []
            for d in range(0, depth + 1):
                if d == depth:
                    parts.append("└─ " if is_last[i] else "├─ ")
                else:
                    has_more = False
                    for j in range(i + 1, n):
                        dj = records[j]["depth"]
                        if dj < d:
                            break
                        if dj == d:
                            has_more = True
                            break
                    parts.append("│  " if has_more else "   ")
            prefixes.append("".join(parts))

        return prefixes

    ########################################
    #           Report rendering           #
    ########################################

    def _aggregate_by_path(self, records: list[dict[str, Any]]):
        """
        Агрегирует записи с одинаковым path (depth-first path от корня).

        Возвращает список агрегированных записей в порядке первого появления.
        Каждая запись: {depth, name, path, meta, avg_time_s, avg_mem_delta,
                        avg_mem_after, avg_mem_peak, count}
        """
        ancestor_names: list[str] = []
        agg: dict[tuple, dict[str, Any]] = {}
        order: list[tuple] = []

        for rec in records:
            d = rec["depth"]
            ancestor_names = ancestor_names[:d]
            path = tuple(ancestor_names + [rec["name"]])
            ancestor_names.append(rec["name"])

            if path not in agg:
                agg[path] = {
                    "depth": d,
                    "name": rec["name"],
                    "path": path,
                    "meta": rec.get("meta", ""),
                    "sum_time": 0.0,
                    "sum_mem_delta": 0.0,
                    "sum_mem_after": 0.0,
                    "sum_mem_peak": 0.0,
                    "count_time": 0,
                    "count_mem": 0,
                }
                order.append(path)

            entry = agg[path]
            if rec.get("time") is not None:
                entry["sum_time"] += rec["time"]
                entry["count_time"] += 1
            if rec.get("mem_delta") is not None:
                entry["sum_mem_delta"] += rec["mem_delta"]
                entry["sum_mem_after"] += rec["mem_after"]
                entry["sum_mem_peak"] = max(
                    entry["sum_mem_peak"],
                    rec["mem_peak"],
                )
                entry["count_mem"] += 1

        result = []
        for path in order:
            e = agg[path]
            result.append(
                {
                    "depth": e["depth"],
                    "name": e["name"],
                    "path": e["path"],
                    "meta": e["meta"],
                    "avg_ms": e["sum_time"] / e["count_time"] * 1000
                    if e["count_time"]
                    else None,
                    "avg_mem_delta": e["sum_mem_delta"] / e["count_mem"]
                    if e["count_mem"]
                    else None,
                    "avg_mem_after": e["sum_mem_after"] / e["count_mem"]
                    if e["count_mem"]
                    else None,
                    "peak_mem": e["sum_mem_peak"] if e["count_mem"] else None,
                },
            )
        return result

    def report(self) -> str:
        """Деревянный отчёт — время и/или память в зависимости от данных."""
        records_raw = [
            r
            for r in self._records
            if r["time"] is not None or r["mem_delta"] is not None
        ]
        if not records_raw:
            return "  [profiler] no data"

        agg = self._aggregate_by_path(records_raw)
        has_time_agg = any(r["avg_ms"] is not None for r in agg)
        has_mem_agg = any(r["avg_mem_delta"] is not None for r in agg)

        lines: list[str] = []

        if has_time_agg:
            lines.extend(self._render_time(agg))

        if has_mem_agg:
            if lines:
                lines.append("")
            lines.extend(self._render_mem(agg))

        return "\n" + "\n".join(lines)

    ########################################
    #          Dot-leader helper           #
    ########################################

    @staticmethod
    def _dot_leader(label: str, leader_col: int, min_dots: int = 1) -> str:
        """
        Возвращает label + пробел + точки до фиксированной колонки
        leader_col.
        """
        dots = max(min_dots, leader_col - len(label) - 1)
        return label + " " + "." * dots

    ########################################
    #            Time rendering            #
    ########################################

    def _render_time(self, agg: list[dict[str, Any]]) -> list[str]:
        time_recs = [r for r in agg if r["avg_ms"] is not None]
        if not time_recs:
            return []

        path_ms: dict[tuple, float] = {
            r["path"]: r["avg_ms"] for r in time_recs
        }
        top_avg_ms = sum(r["avg_ms"] for r in time_recs if r["depth"] == 0)
        prefixes = self._build_tree_layout(time_recs)

        # leader_col: колонка где начинается число (от начала строки с "  ")
        LEADER_COL = 46  # "  " + label + dots → до этой позиции
        # число: "  XX.XXX ms/step  (XX.X%)"  — фиксировано вправо от leader_col
        NUM_W = 8  # ширина поля числа (X.XXX)

        # Вычисляем ширину бокса из самой длинной строки
        # (с процентом от родителя)
        max_pct_len = max(
            len(f"({100.0:>5.1f}% of {'LowRankMultivariateNormal'})"),
            len("(100.0%)"),
        )
        total_w = LEADER_COL + NUM_W + len(" ms/step  ") + max_pct_len + 4
        sep = "  " + "─" * (total_w - 4)

        title = "  Profiler Report  |  time"
        lines: list[str] = []
        lines.append("┌" + "─" * (total_w - 2) + "┐")
        lines.append("│" + title.ljust(total_w - 2) + "│")
        lines.append("└" + "─" * (total_w - 2) + "┘")
        lines.append("  section")
        lines.append(sep)

        for i, rec in enumerate(time_recs):
            label = prefixes[i] + rec["name"]
            if rec["meta"]:
                label += f"  [{rec['meta']}]"

            avg_ms = rec["avg_ms"]
            depth = rec["depth"]

            if depth == 0:
                pct = avg_ms / top_avg_ms * 100 if top_avg_ms > 0 else 0.0
                pct_str = f"({pct:>5.1f}%)"
            else:
                parent_path = rec["path"][:-1]
                parent_ms = path_ms.get(parent_path, 0.0)
                pct = avg_ms / parent_ms * 100 if parent_ms > 0 else 0.0
                parent_name = rec["path"][-2]
                pct_str = f"({pct:>5.1f}% of {parent_name})"

            leader = self._dot_leader(label, LEADER_COL)
            lines.append(f"  {leader} {avg_ms:>{NUM_W}.3f} ms/step  {pct_str}")

        lines.append(sep)
        leader = self._dot_leader("TOTAL (tracked)", LEADER_COL)
        lines.append(f"  {leader} {top_avg_ms:>{NUM_W}.3f} ms/step")
        return lines

    ########################################
    #           Memory rendering           #
    ########################################

    def _render_mem(self, agg: list[dict[str, Any]]) -> list[str]:
        mem_recs = [r for r in agg if r["avg_mem_delta"] is not None]
        if not mem_recs:
            return []

        if torch.cuda.is_available() and self.device is not None:
            peak_mb = torch.cuda.max_memory_allocated(self.device) / 1024**2
        else:
            peak_mb = None

        LEADER_COL = 52
        cols = ["Δ MB", "cur MB", "peak MB"]
        col_widths = [8, 9, 9]
        total_w = LEADER_COL + sum(w + 1 for w in col_widths) + 4

        prefixes = self._build_tree_layout(mem_recs)

        title_parts = ["Profiler Report", "mem"]
        if peak_mb is not None:
            title_parts.append(f"peak={peak_mb:.0f} MB")
        title = "  " + "  |  ".join(title_parts)

        sep = "  " + "─" * (total_w - 4)

        lines: list[str] = []
        lines.append("┌" + "─" * (total_w - 2) + "┐")
        lines.append("│" + title.ljust(total_w - 2) + "│")
        lines.append("└" + "─" * (total_w - 2) + "┘")

        hdr = f"  {'section'}"
        for c, w in zip(cols, col_widths, strict=False):
            hdr += f"  {c:>{w}}"
        lines.append(hdr)
        lines.append(sep)

        for i, rec in enumerate(mem_recs):
            label = prefixes[i] + rec["name"]
            if rec["meta"]:
                label += f"  [{rec['meta']}]"

            delta_str = f"{rec['avg_mem_delta']:+.0f}"
            cur_str = f"{rec['avg_mem_after']:.0f}"
            peak_str = f"{rec['peak_mem']:.0f}"

            leader = self._dot_leader(label, LEADER_COL)
            lines.append(
                f"  {leader}"
                f" {delta_str:>{col_widths[0]}}"
                f" {cur_str:>{col_widths[1]}}"
                f" {peak_str:>{col_widths[2]}}",
            )

        lines.append(sep)
        return lines

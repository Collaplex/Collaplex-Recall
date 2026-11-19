#!/usr/bin/env python3
"""M1 词汇鸿沟冒烟：对比别名展开前后的 top1/top5。

前提：fixtures/notes 已入库，NOTE_EMBED 等环境变量指向同一个库和模型
（见 README 的 llama.cpp 实测一节）。脚本会幂等补几条测试别名。
"""
import json
import os
import subprocess

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NOTE = os.path.join(PROJ, "target", "debug", "note")

ALIASES = [
    ("K7", "PipeWire 虚拟声卡"),
    ("B2", "RTX 5060"),
    ("断网", "掉线"),
    ("小破卡", "显卡"),
]

QUERIES = [
    ("K7 是什么", "音频链路.md"),
    ("B2 是什么", "显卡与模型.md"),
    ("K7 和 B2 有关系吗", "音频链路.md"),
    ("断网是什么", "校园网认证排查.md"),
    ("显存断崖在哪", "显卡与模型.md"),
    ("模型能改笔记吗", "笔记软件架构.md"),
]


def run(no_alias: bool):
    top1 = top5 = 0
    rows = []
    for q, expect in QUERIES:
        args = [NOTE, "search", q, "-k", "5", "--json"]
        if no_alias:
            args.append("--no-alias")
        out = subprocess.run(args, capture_output=True, text=True, cwd=PROJ)
        data = json.loads(out.stdout)
        paths = [h["path"] for h in data["hits"]]
        rank = paths.index(expect) + 1 if expect in paths else 0
        rows.append((q, rank))
        top1 += rank == 1
        top5 += rank > 0
    return top1, top5, rows


def main():
    for alias, entity in ALIASES:
        subprocess.run([NOTE, "alias", "add", alias, entity], capture_output=True)

    for label, no_alias in [("no-alias", True), ("alias", False)]:
        top1, top5, rows = run(no_alias)
        print(f"[{label}] top1={top1}/{len(QUERIES)} top5={top5}/{len(QUERIES)}")
        for q, rank in rows:
            print(f"   {rank if rank else 'MISS':>4} {q}")


if __name__ == "__main__":
    main()

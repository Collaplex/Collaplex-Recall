#!/usr/bin/env python3
"""从真实笔记库抽样生成评测问题（合成集，需人工抽查）。

用法：
  python3 scripts/gen_real_eval.py --db note_real --out data/real/eval --n 50

输出 queries.jsonl / qrels.jsonl，配合：
  recall eval queries.jsonl qrels.jsonl --prefix "" -k 10

注意：这是模型生成的题，不等于人工标注；只用于冒烟和调参。
"""
import argparse
import json
import os
import random
import re
import urllib.request

import psycopg2


def chat(base, model, system, user):
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.4,
            "max_tokens": 120,
            "stream": False,
        }
    ).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return data["choices"][0]["message"]["content"]


def parse_question(text, banned):
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    q = (obj.get("question") or "").strip()
    if not q or len(q) < 8 or len(q) > 120:
        return None
    low = q.lower()
    for b in banned:
        if b and b.lower() in low:
            return None
    for vague in ["这个", "这段", "这篇文章", "该文", "本文", "上述", "它", "该插件", "该"]:
        if vague in q:
            return None
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="note_real")
    ap.add_argument("--out", default="data/real/eval")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--base", default="http://127.0.0.1:8082/v1")
    ap.add_argument("--model", default="local-chat")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    conn = psycopg2.connect(
        dbname=args.db, user=os.environ.get("USER", "postgres"), host=os.path.expanduser("~/pgdata/socket")
    )
    cur = conn.cursor()
    cur.execute(
        "select n.path, c.heading, c.text from chunks c join notes n on n.id=c.note_id "
        "where length(c.text) between 150 and 1200 order by random() limit %s",
        (args.n * 6,),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    random.Random(args.seed).shuffle(rows)

    system = (
        "你是评测题生成器。根据给定笔记片段写一个用户会问的问题。"
        "要求：问题必须能由片段回答；必须自足，不能出现「这个/这段/这篇文章/它/本文」"
        "这类没有上下文的指代；不要出现文件名、路径或标题原文；"
        "不要问日期、作者、来源链接等元数据；不要问需要外部知识或推断的内容；"
        "只输出 JSON：{\"question\":\"...\"}。"
    )
    os.makedirs(args.out, exist_ok=True)
    qpath = os.path.join(args.out, "queries.jsonl")
    rpath = os.path.join(args.out, "qrels.jsonl")
    made = 0
    with open(qpath, "w", encoding="utf-8") as qf, open(
        rpath, "w", encoding="utf-8"
    ) as rf:
        for path, heading, text in rows:
            if made >= args.n:
                break
            low_head = heading.lower()
            toc_head = any(
                x in low_head for x in ("∑", "目录", "导览", "索引", "moc", "readme")
            )
            if (
                text.count("](") >= 3
                or text.count("[[") >= 3
                or toc_head
            ):
                continue
            banned = [os.path.basename(path), os.path.splitext(os.path.basename(path))[0]]
            try:
                out = chat(args.base, args.model, system, f"标题：{heading}\n片段：{text[:800]}")
                q = parse_question(out, banned)
            except Exception as e:
                print("skip", path, e)
                continue
            if not q:
                continue
            made += 1
            qid = f"q{made}"
            qf.write(json.dumps({"id": qid, "text": q}, ensure_ascii=False) + "\n")
            rf.write(
                json.dumps({"qid": qid, "pid": path, "score": 1}, ensure_ascii=False)
                + "\n"
            )
    print(f"生成 {made} 条问题：{qpath}")


if __name__ == "__main__":
    main()

# 우선 rag만 text2sql 제외하구..;;

from typing import List, Dict

def merge_and_rank_results(rag_results, sql_results, top_k=15):
    merged = {}
    duplicate_ids = set()

    for result in rag_results:
        pid = result["정책ID"]
        if pid not in merged:
            merged[pid] = result
        else:
            merged[pid]["점수"] = max(merged[pid]["점수"], result["점수"])

    for result in sql_results:
        pid = result["정책ID"]
        if pid in merged:
            duplicate_ids.add(pid)
            merged[pid]["점수"] = max(merged[pid]["점수"], result["점수"])
        else:
            merged[pid] = result

    sorted_duplicates = sorted(
        [v for k, v in merged.items() if k in duplicate_ids],
        key=lambda x: x["점수"],
        reverse=True
    )

    sorted_non_duplicates = sorted(
        [v for k, v in merged.items() if k not in duplicate_ids],
        key=lambda x: x["점수"],
        reverse=True
    )

    return (sorted_duplicates + sorted_non_duplicates)[:top_k]

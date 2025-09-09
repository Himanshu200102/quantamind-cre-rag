from typing import List, Dict, Any

def topk(chunks: List[Dict[str, Any]], k: int) -> List[Dict[str, Any]]:
    return sorted(chunks, key=lambda x: x["score"], reverse=True)[:k]

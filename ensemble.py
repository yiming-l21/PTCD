
from typing import List, Literal, Optional
def _majority_vote(cands: List[str], label_space: List[str]) -> str:
    from collections import Counter
    ctr = Counter([c for c in cands if isinstance(c, str) and c])
    if not ctr:
        return label_space[0]
    winners = ctr.most_common()
    top = winners[0][1]
    ties = [lab for lab, c in winners if c == top]
    if len(ties) == 1:
        return ties[0]
    lows = [t.lower() for t in ties]
    for lab in label_space:
        if lab.lower() in lows:
            return lab
    return label_space[0]

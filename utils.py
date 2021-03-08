from typing import *
import pickle


def load_vocab(filename: str) -> Tuple[Dict[int, str], List[int], Dict[int, int]]:
    eid2name = {}
    keywords = []
    with open(filename, encoding='utf-8') as f:
        for line in f:
            temp = line.strip().split('\t')
            eid = int(temp[1])
            eid2name[eid] = temp[0]
            keywords.append(eid)
    eid2idx = {w: i for i, w in enumerate(keywords)}  # eid : index
    print(f'Vocabulary: {len(keywords)} keywords loaded')
    # eid2name: [dict] entity 的 eid 对应 entity的name
    # keywords: 纯eid
    # eid2idx： [dict] eid : index
    return eid2name, keywords, eid2idx


def load_entity_pos(self, filename):
    return pickle.load(open(filename), 'rb')

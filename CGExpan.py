from typing import *
import os
import torch
from transformers import BertTokenizer, BertModel, BertForMaskedLM
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as cos
import random
import math
from collections import defaultdict as ddict
from entitys import Entity
import queue
import copy
import nltk
import inflect
from datareader import DataReader
import time

GENERATION_SAMPLE_SIZE = 6
EXPANSION_SAMPLE_SIZE = 3
POS_CNAME_THRES = 5. / 6
TOKEN_PATH = '/home/amax/ZWX/bert_data/bert_base_uncase/bert-base-uncased-vocab.txt'
BERT_PATH = '/home/amax/ZWX/bert_data/bert_base_uncase/'


class CGExpan(object):

    def __init__(self,
                 device,
                 reader: DataReader,
                 k=5,
                 gen_thres=3,
                 model_name='bert-base-uncased',
                 ):
        self.tokenizer = BertTokenizer.from_pretrained(TOKEN_PATH, do_lower_case=False)
        self.maskedLM = BertForMaskedLM.from_pretrained(BERT_PATH, output_hidden_states=True)

        self.maskedLM.to(device)
        self.maskedLM.eval()

        self.k = k  # TODO
        self.gen_thres = gen_thres  # TODO

        self.reader = reader
        self.eid2name = reader.get_eid2name()
        self.keywords = reader.get_keywords()
        self.eid2idx = reader.get_eid2idx()  # TODO
        self.entity_pos = reader.get_entity_pos()  # TODO
        self.pretrained_emb = reader.get_pretrained_emb()  # TODO

        self.means = np.array([np.mean(emb, axis=0) for emb in self.get_emb_iter()])

        self.inflect = inflect.engine()

        mask_token = self.tokenizer.mask_token

        self.generation_templates = [
            [mask_token, ' such as {} , {} , and {} .', 1],
            ['such ' + mask_token, ' as {} , {} , and {} .', 1],
            ['{} , {} , {} or other ' + mask_token, ' .', 0],
            ['{} , {} , {} and other ' + mask_token, ' .', 0],
            [mask_token, ' including {} , {} , and {} .', 1],
            [mask_token, ' , especially {} , {} , and {} .', 1],
        ]

        self.ranking_templates = [
            '{} such as ' + mask_token + ' .',
            'such {} as ' + mask_token + ' .',
            mask_token + ' or other {} .',
            mask_token + ' and other {} .',
            '{} including ' + mask_token + ' .',
            '{} especially ' + mask_token + ' .',
        ]

        self.expansion_templates = [
            ('', ' such as {} , {} , {} , and {} .'),
            ('such ', ' as {} , {} , {} , and {} .'),
            ('{} , {} , {} , {} or other ', ' .'),
            ('{} , {} , {} , {} and other ', ' .'),
            ('', ' including {} , {} , {} , and {} .'),
            ('', ' , especially {} , {} , {} , and {} .'),
        ]

        self.calculated_cname_rep = {}

        print(f'<CGExpan>: CGExpan is ready!')

    def rand_idx(self, l):
        # ?????????l??????????????????????????????
        for _ in range(10000):
            for i in np.random.permutation(l):
                yield i

    def get_emb(self, i):
        return self.pretrained_emb[self.entity_pos[i]:self.entity_pos[i + 1]]

    def get_emb_iter(self):  # ????????????iterable????????????
        for i in range(len(self.keywords)):
            yield self.pretrained_emb[self.entity_pos[i]:self.entity_pos[i + 1]]

    def class_name_generation(self, set_text):
        cname2count = ddict(int)
        idx_generator = self.rand_idx(len(set_text))
        # set_text ????????????????????????entity???str?????????eid?????????
        for _ in range(GENERATION_SAMPLE_SIZE):
            for template in self.generation_templates:
                candidate = set()
                q = queue.Queue()
                q.put([])

                template = copy.deepcopy(template)  # ?????????pattern????????????????????????self.generation_templates
                indices = []
                # ?????????????????????idx_generator??????????????????words?????????pattern????????????
                for n in idx_generator:
                    if n not in indices:
                        indices.append(n)
                        if len(indices) == 3:
                            break
                template[template[2]] = template[template[2]].format(*[set_text[i] for i in indices])
                # template[2] ???mask_token?????????????????????pattern???????????????

                while not q.empty():
                    # ??????q????????????c????????????
                    # ?????????c??????[],?????????countries, cities, etc.
                    # ????????????predict??????classname?????? something like adj+countries???Asian countries)
                    # ??????????????????classname.
                    c = q.get()
                    if len(c) >= 2:
                        continue
                    text = template[0] + (' ' if len(c) > 0 else '') + ' '.join(c) + template[1]  # query
                    ids = torch.tensor([self.tokenizer.encode(text, max_length=512)]).long()
                    mask_pos = (ids == self.tokenizer.mask_token_id).nonzero()[0, 1]
                    ids = ids.cuda()

                    with torch.no_grad():
                        predictions = self.maskedLM(ids)[0]
                    _, predicted_index = torch.topk(predictions[0, mask_pos], k=3)
                    predicted_token = [self.tokenizer.convert_ids_to_tokens([idx.item()])[0] for idx in predicted_index]

                    for t in predicted_token:
                        # nltk.pos_tag????????????
                        tag = nltk.pos_tag([t] + c)
                        tag = tag[0][1]
                        # JJ ???????????? NNS ?????????????????????NN ??????????????????
                        if tag in set(['JJ', 'NNS', 'NN']) and t not in set(c) \
                                and t not in set([self.inflect.plural(cc) for cc in c]) and t not in ['other', 'such',
                                                                                                      'others']:
                            if len(c) == 0 and tag == 'JJ':
                                continue
                            if len(c) == 0 and tag == 'NN':
                                t = self.inflect.plural(t)  # ??????????????????
                            new = [t] + c
                            candidate.add(tuple(new))
                            q.put(new)
                for c in candidate:
                    cname2count[' '.join(c)] += 1  # ?????????class name ???predict????????????
        return cname2count

    def class_name_ranking(self, cname2count, query_set, expanded_set, neg_cnames, prev_cn, margin):
        current_set = query_set + expanded_set
        cnames = [cname for cname in cname2count if cname2count[cname] >= self.gen_thres]
        cnames += [cn for cn in prev_cn if cn not in cnames]
        cname2idx = {cname: i for i, cname in enumerate(cnames)}
        cnames_rep = np.vstack([self.get_cname_rep(cname) for cname in cnames])
        scores = np.zeros((len(current_set), len(cnames)))

        # ????????????entity?????????cname????????????score
        for i, eid in enumerate(current_set):
            emb = self.get_emb(self.eid2idx[eid])
            if len(emb) < self.k:
                continue
            sims = cos(cnames_rep, emb)  # ???????????????
            for j in range(len(cnames)):
                # ???i???entity?????????j???cname?????????score
                scores[i, j] = np.mean(np.partition(np.amax(sims[j * 6:(j + 1) * 6], axis=0), -self.k)[-self.k:])

        cname2mrr = ddict(float)

        # ??????entity????????????class name????????????score???
        # score?????????cname??????1???????????????/2???????????????/3??????
        for eid, score in zip(current_set, scores):
            r = 0.
            for i in np.argsort(-score):
                cname = cnames[i]
                if cname2count[cname] < min(GENERATION_SAMPLE_SIZE * len(self.generation_templates) * POS_CNAME_THRES,
                                            max(cname2count.values())) and cname not in prev_cn:
                    continue
                r += 1
                cname2mrr[cname] += 1 / r
        pos_cname = sorted(cname2mrr.keys(), key=lambda x: cname2mrr[x], reverse=True)[0]
        # ???????????????cname??????positive cname

        # find negative entities
        uni_cnames = [cname for cname in cnames if len(cname.split(' ')) == 1 and not pos_cname.endswith(cname)]
        this_neg_cnames = set(uni_cnames)
        for eid, score in zip(query_set, scores):
            ranked_uni_cnames = sorted([pos_cname] + uni_cnames, key=lambda x: score[cname2idx[x]], reverse=True)
            for i, cname in enumerate(ranked_uni_cnames):
                if cname == pos_cname:
                    break
            this_neg_cnames = this_neg_cnames & set(ranked_uni_cnames[i + 1 + margin:])

        # ??????positive???cname???negative???cnames
        return pos_cname, neg_cnames | this_neg_cnames  # ??????

    def class_guided_expansion(self, pos_cname, current_set, set_text, neg_set):
        global_idx_generator = self.rand_idx(len(current_set))
        local_idx_generator = self.rand_idx(len(current_set))
        global_scores = cos(self.means[[self.eid2idx[eid] for eid in current_set]], self.means)

        # 1.???pattern???entity???????????????????????????
        ids = []
        for _ in range(EXPANSION_SAMPLE_SIZE):
            for template in self.expansion_templates:
                indices = []  # ????????????entity
                for n in local_idx_generator:
                    if n not in indices:
                        indices.append(n)
                        if len(indices) == 3:
                            break
                fill_in = [self.tokenizer.mask_token] + [set_text[i] for i in indices]
                fill_in = np.random.permutation(fill_in)
                # fill_in: ['MASK','entity1_name','entity2_name','entity3_name']??????
                text = template[0] + pos_cname + template[1]
                text = text.format(*fill_in)
                ids.append(self.tokenizer.encode(text, max_length=512))

        mask_rep = self.get_mask_rep(ids)  # mask_rep????????????entity?????????

        eid2mrr = ddict(float)
        for local_rep in mask_rep:
            indices = []
            for n in global_idx_generator:
                if n not in indices:
                    indices.append(n)
                    if len(indices) == 3:
                        break
            this_global_score = np.mean(global_scores[indices], axis=0)
            this_global_score_ranking = np.argsort(-this_global_score)

            # keywords ???eid?????????
            this_keywords = [self.keywords[i] for i in this_global_score_ranking[:500]]  # eid
            this_global_score = [this_global_score[i] for i in this_global_score_ranking[:500]]
            this_embs = [self.get_emb(i) for i in [self.eid2idx[eid] for eid in this_keywords]]  # eid->index->embedding
            this_entity_pos = [0] + list(np.cumsum([len(emb) for emb in this_embs]))
            this_embs = np.vstack(this_embs)

            raw_local_scores = cos(local_rep[np.newaxis, :], this_embs)[0]

            local_scores = np.zeros((500,))
            for i in range(500):
                start_pos = this_entity_pos[i]
                end_pos = this_entity_pos[i + 1]
                if end_pos - start_pos < self.k:
                    local_scores[i] = 1e-8
                else:
                    local_scores[i] = np.mean(np.partition(raw_local_scores[start_pos:end_pos], -self.k)[-self.k:])

            scores = 5 * np.log(local_scores) + np.log(this_global_score)

            r = 0.
            for i in np.argsort(-scores):
                eid = this_keywords[i]
                if eid not in set(current_set) and eid not in neg_set:
                    r += 1
                    eid2mrr[eid] += 1 / r
                if r >= 20:
                    break

        eid_rank = sorted(eid2mrr, key=lambda x: eid2mrr[x], reverse=True)
        for i, eid in enumerate(eid_rank):
            if eid2mrr[eid] < EXPANSION_SAMPLE_SIZE * len(self.expansion_templates) * 0.2:
                break
        return eid_rank[:max(15, i)]  # 10 candidates for one round
        # return eid_rank[:max(5, i)]  # five candidates for one round

    def expand(self, seeds: List[Entity], margin: int = 2) -> List[Entity]:  # TODO!

        # TODO: List(Entity) -> List(int)
        query_set = [seed.eid for seed in seeds]
        print('start expanding: ' + str([self.eid2name[eid] for eid in query_set]))
        start_time = time.time()

        expanded_set = []
        prev_cn = set()  # useless?
        neg_set = set()
        neg_cnames = set()

        # 1.generate class names

        set_text = [self.eid2name[eid].lower() for eid in query_set + expanded_set]

        cname2count = self.class_name_generation(set_text)

        # 2.class name ranking

        pos_cname, neg_cnames = self.class_name_ranking(cname2count, query_set, expanded_set, neg_cnames, prev_cn, margin)
        prev_cn.add(pos_cname)  # useless?

        # 3.expansion

        new_entities = self.class_guided_expansion(pos_cname, query_set + expanded_set, set_text, neg_set)

        expanded_set.extend(new_entities)
        # TODO!
        # expanded_set: List[int] -> List(Entity)
        expanded_set = self.reader.convert_eid_2_entity(expanded_set)

        print(f'num of expanded entities: {len(expanded_set)}, '
              f'time: {int((time.time() - start_time) / 60)} min {int(time.time() - start_time) % 60} sec')

        return expanded_set

    def get_cname_rep(self, cname):
        # ??????cname???BERT???????????????
        if cname not in self.calculated_cname_rep:
            ids = []
            for template in self.ranking_templates:
                text = copy.deepcopy(template).format(cname)
                ids.append(self.tokenizer.encode(text, max_length=512))
                # tokenizer??????????????????????????????????????????????????????id
                # ?????? encode() ??????????????? ?????? tokenization ?????? token ???????????? id ???????????????????????? Bert ?????????
            self.calculated_cname_rep[cname] = self.get_mask_rep(ids)
        return self.calculated_cname_rep[cname]

    def get_mask_rep(self, batch_ids):  # ??????ids???????????????BERT???????????????
        # ?????????batch_ids???????????????????????????
        # e.g. [[1,1,1],[2,1]] -> [[1,1,1,0,0],[2,1,0,0,0]]
        batch_max_length = max(len(ids) for ids in batch_ids)
        ids = torch.tensor([ids + [0 for _ in range(batch_max_length - len(ids))] for ids in batch_ids]).long()
        masks = (ids != 0).long()
        temp = (ids == self.tokenizer.mask_token_id).nonzero()  # [MASK]???????????????????????????0
        mask_pos = []  # ??????mask_token?????????????????????

        for ti, t in enumerate(temp):
            # nonzero????????????????????????????????????mask????????????ti???index???t??????????????????t???????????????x,y????????????
            # x -- ???????????????
            # y -- ???????????????????????????
            assert t[0].item() == ti  # index ?????? ??????
            mask_pos.append(t[1].item())

        ids = ids.to('cuda')
        masks = masks.to('cuda')
        with torch.no_grad():
            batch_final_layer = self.maskedLM(ids, masks)[1][-1]

        # ??????mask_token???BERT???????????????????????????
        return np.array([final_layer[idx].cpu().numpy() for final_layer, idx in zip(batch_final_layer, mask_pos)])

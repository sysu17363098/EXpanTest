from typing import *
import torch
import numpy as np
from entitys import Entity
from utils import load_vocab
import os
import pickle

data_info = {
    'wiki': {
        'path': '/home/amax/ZWX/data/wiki/',
        'vocab': 'entity2id.txt',
        'sentences': 'sentences.json',
        'emb_file': 'pretrained_emb.npy',
        'entity_pos': 'entity_pos.pkl',
    },
    'APR': {
        'path': '/home/amax/ZWX/data/APR/',
        'vocab': 'entity2id.txt',
        'sentences': 'sentences.json',
        'emb_file': 'pretrained_emb.npy',
        'entity_pos': 'entity_pos.pkl',
    }
}


class DataReader:

    def __init__(self, dataset: str, field: str):
        """
        :param dataset: the name of the dataset you decide to use
        :param field:  the field/subject/class to which the seeds belong
        """

        # init:
        # self.eid2name: Dict[eid(int):name(str)]
        # self.keywords: List[eid(int)]
        # self.eid2idx: Dict[eid(int):idx(int)]
        # self.entity_pos: List[entity_position(int)] record where the vectors of an entity start in self.pretrained_emb
        # self.pretrained_emb: memmap, pre-trained word embeddings
        # self.original_seeds: List[Entity] original seed set
        # self.gt: List[Entity] ground-truth entity set

        if dataset == 'wiki' or dataset == 'APR':
            print(f'<DataReader>: data set:{dataset},field:{field}.')
            data_path = data_info[dataset]['path']

            # load vocabulary
            print("loading vocabulary...")
            vocab = os.path.join(data_path, data_info[dataset]['vocab'])
            self.eid2name, self.keywords, self.eid2idx = load_vocab(vocab)

            # load entity position
            print("loading entity position...")
            entity_pos_out = os.path.join(data_path, data_info[dataset]['entity_pos'])
            self.entity_pos = pickle.load(open(entity_pos_out, 'rb'))

            # load pre-trained embedding
            print("loading pre-trained embedding...")
            emb_file = os.path.join(data_path, data_info[dataset]['emb_file'])
            self.pretrained_emb = np.memmap(emb_file, dtype='float32', mode='r', shape=(self.entity_pos[-1], 768))

            # read seeds in a certain field
            if field != 'all':
                # load seeds from folder:query
                print("loading seeds...")
                self.original_seeds = []
                with open(os.path.join(data_path, 'query', field+'.txt'), encoding='utf-8') as f:
                    for line in f:
                        if line == 'EXIT\n': break
                        temp = line.strip().split(' ')
                        eids = [int(eid) for eid in temp]
                        self.original_seeds.extend(self.convert_eid_2_entity(eids))

                for seed in self.original_seeds:  # seeds in our consider is ground-truth
                    seed.ground_truth = True

                # load ground truth set from folder:gt
                print("loading gt...")
                self.gt = []
                with open(os.path.join(data_path, 'gt', field+'.txt'), encoding='utf-8') as f:
                    for line in f:
                        temp = line.strip().split('\t')
                        eid = int(temp[0])
                        if int(temp[2]) >= 1:
                            self.gt.extend(self.convert_eid_2_entity([eid]))

                for i in self.gt:
                    i.ground_truth = True

            else:  # if we need to read all files
                for file in os.listdir(os.path.join(data_path, 'query')):
                    # load seeds from folder:query
                    print("loading seeds...")
                    self.original_seeds = []
                    with open(os.path.join(data_path, 'query', file), encoding='utf-8') as f:
                        for line in f:
                            if line == 'EXIT\n': break
                            temp = line.strip().split(' ')
                            eids = [int(eid) for eid in temp]
                            self.original_seeds.extend(self.convert_eid_2_entity(eids))

                    # load ground truth set from folder:gt
                    print("loading gt...")
                    self.gt = []
                    with open(os.path.join(data_path, 'gt', file), encoding='utf-8') as f:
                        for line in f:
                            temp = line.strip().split('\t')
                            eid = int(temp[0])
                            if int(temp[2]) >= 1:
                                self.gt.extend(self.convert_eid_2_entity([eid]))
            print("<DataReader>: Completed.\n\n")

    def get_eid2name(self) -> Dict[int, str]:
        return self.eid2name

    def get_keywords(self) -> List[int]:
        return self.keywords

    def get_eid2idx(self) -> Dict[int, int]:
        return self.eid2idx

    def get_entity_pos(self):
        return self.entity_pos

    def get_pretrained_emb(self):
        return self.pretrained_emb

    def convert_eid_2_entity(self, eids: List[int]) -> List[Entity]:
        entities = []
        for eid in eids:
            name = self.eid2name[eid]
            i = self.eid2idx[eid]
            vectors = self.pretrained_emb[self.entity_pos[i]:self.entity_pos[i + 1]]
            vector = torch.from_numpy(np.mean(vectors, axis=0))
            entities.append(Entity(name=[name], vector=vector, ground_truth=None, eid=eid))
        return entities

    def get_original_seeds(self) -> List[Entity]:
        return self.original_seeds

    def get_gt_set(self) -> List[Entity]:
        return self.gt


if __name__ == '__main__':
    reader = DataReader('wiki', 'countries')
    print('eid2idx', len(reader.get_eid2idx()))
    print('eid2name', len(reader.get_eid2name()))
    print('keywords', reader.get_keywords())
    print('original_seeds', reader.get_original_seeds())
    print('gt_set', reader.get_gt_set())
    print('vector', reader.get_gt_set()[0].vector.shape)
    print('vector', reader.get_gt_set()[1].vector)
    print('test', reader.convert_eid_2_entity([131078])[0].vector.shape)


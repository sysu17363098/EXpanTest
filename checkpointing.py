import os
import uuid
import json
import time
import torch
import logging

logger = logging.getLogger()


class Checkpointing:

    def __init__(self, directory, keep_n=10, metric_bigger_better=False):
        if not os.path.exists(directory):
            os.makedirs(directory)
        self.base_dir = directory
        self.path_to_cp2info = os.path.join(self.base_dir, "cp2metric.json")
        if not os.path.exists(self.path_to_cp2info):
            self.set_json(self.path_to_cp2info, [])

        self.keep_n = keep_n
        self.metric_bigger_better = metric_bigger_better

    def get_json(self, path):
        with open(path, mode="r", encoding="utf-8") as f:
            return json.load(f)

    def set_json(self, path, d):
        with open(path, mode="w", encoding="utf-8") as f:
            return json.dump(d, f, ensure_ascii=False, indent=2)

    def save_check_point(self, dict, metric):
        file_name = time.strftime("%Y%m%d_%H_%M_%S_%a") + "_%s" % uuid.uuid1().hex
        torch.save(dict, os.path.join(self.base_dir, file_name))

        record: list = self.get_json(self.path_to_cp2info)
        record.append((file_name, metric, time.time()))
        record.sort(key=lambda x: x[2], reverse=True)
        if len(record) >= self.keep_n:
            deleted = record[self.keep_n:]
            for f, score, *_ in deleted:
                logging.info("Deleting oldest checkpoint: %s" % f)
                os.remove(os.path.join(self.base_dir, f))
            record = record[0:self.keep_n]

        self.set_json(self.path_to_cp2info, record)

    def load_check_point(self, best=True, recent=False):
        record: list = self.get_json(self.path_to_cp2info)
        if len(record) == 0:
            logger.info("%s has no checkpoints recorded." % self.base_dir)
            return None
        if best:
            record.sort(key=lambda x: x[1], reverse=self.metric_bigger_better)
        elif recent:
            record.sort(key=lambda x: x[2], reverse=True)

        p = os.path.join(self.base_dir, record[0][0])
        logger.info(f"Loading parameters from {p}")
        return torch.load(p, map_location='cuda:0')

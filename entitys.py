from typing import *
from dataclasses import dataclass
import torch


@dataclass
class Entity:
    name: Optional[List[str]]
    vector: Optional[torch.tensor]  # (word_vec_dim,)
    ground_truth: Optional[bool]
    eid: int

    def __hash__(self):
        return hash(self.eid)

    def __eq__(self, other):
        return self.eid == other.eid

    def __repr__(self):
        return f'<Entity {(self.name, self.eid, self.ground_truth)})>'


if __name__ == '__main__':
    x = torch.arange(12).view(12,)
    anEntity = Entity(['Rocket'], x, True, 12)
    print(anEntity)

    l = [anEntity]
    anotherEntity = Entity(['Rocket2'], x, True, 12)
    if anotherEntity in l:
        print("True")
    else:
        print("False")


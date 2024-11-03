from typing import Dict, List

import torch
import torch.nn as nn


class PPT(nn.Module):
    def __init__(self, backbone, conditions: List[str], context_channels: int = 256):
        super().__init__()
        self.backbone = backbone()
        self.conditions = conditions

        self.embedding_table = nn.Embedding(len(conditions), context_channels)

    def forward(self, batch_dict: Dict):
        condition = batch_dict["condition"][0]
        assert condition in self.conditions
        context = self.embedding_table(
            torch.tensor([self.conditions.index(condition)], device=batch_dict["coord"].device)
        )
        batch_dict["context"] = context
        return self.backbone(batch_dict)

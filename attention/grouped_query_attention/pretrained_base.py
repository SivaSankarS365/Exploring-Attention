from imports import *
from attention import BaseMultiHeadedAttention


class BasePreTrainedGroupedQueryAttention(BaseMultiHeadedAttention):
    """
    Modifies *trained* Multi-Headed Attention to Grouped Query Attention as done in https://arxiv.org/pdf/2305.13245v3.pdf
    """
    def __init__(self,embed_dim,num_heads,num_groups):
        super().__init__(embed_dim,num_heads)
        self.num_groups = num_groups
        assert num_heads%num_groups==0, "num_heads must be divisible by num_groups"

    def _group(self,x):
        """
        Grouping with mean pooling as suggested by https://arxiv.org/pdf/2305.13245v3.pdf

        key,value shape: B,H,S,E => B,G,H//G,S,E ===mean pooling===> B,G,S,E
        To ensure order is correct, we permute to B,S,E,H then group to B,S,E,G,H//G
        Then mean pool B,S,E,G,H//G => B,S,E,G ===permute===> B,G,S,E
        Then Interleave repeat to B,H,S,E
        """

        B,H,S,E = x.shape
        G = self.num_groups

        x = x.permute([0,2,3,1]) # B,S,E,H
        x = x.reshape(B,S,E,G,H//G) # B,S,E,G,H//G
        x = x.mean(dim=-1) # B,S,E,G
        x = x.permute([0,3,1,2]) # B,G,S,E
        x = torch.repeat_interleave(x,H//G,dim=1)
        return x

    def construct_query_key_value(self, x):
        query,key,value =  super().construct_query_key_value(x)

        B,H,S,E = key.shape
        G = self.num_groups

        key = self._group(key)
        value = self._group(value)

        return query,key,value
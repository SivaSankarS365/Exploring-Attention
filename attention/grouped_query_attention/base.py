from imports import *
from attention import BaseMultiHeadedAttention

class BaseGroupedQueryAttention(BaseMultiHeadedAttention):
    """
    Implementation for training form scratch, 
    directly project to grouped query instead of mean pooling like done in Mistral
    """
    def __init__(self,embed_dim,num_heads,num_groups):
        self.num_groups = num_groups
        super().__init__(embed_dim,num_heads)
        assert num_heads%num_groups==0, "num_heads must be divisible by num_groups"

    def init_qkvo_proj(self):
        kv_head_embed_dim = self.num_groups * (self.embed_dim//self.num_heads)
        self.query_proj = nn.Linear(self.embed_dim,self.embed_dim)

        self.key_proj = nn.Linear(self.embed_dim,kv_head_embed_dim)
        self.value_proj = nn.Linear(self.embed_dim,kv_head_embed_dim)

        self.output_proj = nn.Linear(self.embed_dim,self.embed_dim)


    def construct_query_key_value(self, x,kv_cache=None):
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
     
        query = self._split_head(query,self.num_heads)
        key = self._split_head(key,self.num_groups)
        value = self._split_head(value,self.num_groups)

        key = torch.repeat_interleave(key,self.num_heads//self.num_groups,dim=1)
        value = torch.repeat_interleave(value,self.num_heads//self.num_groups,dim=1)

        return query,key,value
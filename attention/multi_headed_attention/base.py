from imports import *
from utils.data_containers import BaseOutput
from ..self_attention import BaseSelfAttention


class BaseMultiHeadedAttention(BaseSelfAttention):
    """
    Extends self attention to multi-headed attention
    """
    def __init__(self,embed_dim,num_heads):
        super().__init__(embed_dim)
        self.num_heads = num_heads
    
    def create_heads(self,query,key,value):
        B,S,E = query.shape
        query = query.view(B,self.num_heads,S,E//self.num_heads)
        key = key.view(B,self.num_heads,S,E//self.num_heads)
        value = value.view(B,self.num_heads,S,E//self.num_heads)
        return query,key,value
    
        
    def construct_query_key_value(self,x,kv_cache):
        query,key,value = super().construct_query_key_value(x,kv_cache)
        return self.create_heads(query,key,value)
    
    def calculate_unmasked_attention_logits(self,query,key):
        # Q.K' : [B,H, S,E] @ [B,H,E,S]
        key_t = key.transpose(2,3) # Transpose to [B,H,E,S] by exchanging dim 1 and 2
        
        # scaling factor
        scale = math.sqrt(self.embed_dim)

        # Calculate logits
        unmasked_attention_logits = (query@key_t)/scale

        return unmasked_attention_logits
    
    def apply_causal_mask(self,attention_logits,mask_value=None):
        # lower trianglular matrix with 1s
        B,H,Sq,Sk = attention_logits.shape 

        device = attention_logits.device
        causal_mask = torch.tril(torch.ones(B,H,Sk,Sk)).to(device)
        causal_mask = causal_mask[:,:,-Sq:,-Sk:] # Trim off, for kv_cache, no-op if kv_cache is None

        if mask_value is None:
            mask_value = -torch.inf

        # replace upper triangular with -inf or a very large negative number, causal masking!
        masked_attention_logits = attention_logits.masked_fill_(causal_mask == 0, mask_value) 
        return masked_attention_logits
    
    def apply_attention_mask(self,attention_mask,masked_attention_logits):
        masked_attention_logits = masked_attention_logits.masked_fill_(
            attention_mask[:,None,None,:] == 0, # Additional dimension for heads
            -torch.inf)
        return masked_attention_logits
    

    
    def calculate_final_output(self,attention_weights,value):
        B,H,S,E = attention_weights.shape
        attention_output = attention_weights@value

        attention_output = attention_output.view(B,S,E*H) # flatten back to (B,S,E)

        final_output = self.output_proj(attention_output)
        return final_output
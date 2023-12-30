from imports import *
from utils.data_containers import BaseOutput

class BaseSelfAttention(nn.Module):
    """
    Minimal Base Self Attention
    """
    def __init__(self,embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.init_qkvo_proj()

    def init_qkvo_proj(self):
        self.query_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.key_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.value_proj = nn.Linear(self.embed_dim,self.embed_dim)
        self.output_proj = nn.Linear(self.embed_dim,self.embed_dim)

    def construct_query_key_value(self,x):
        # Construct Q, K, V
        query = self.query_proj(x)
        key = self.key_proj(x)
        value = self.value_proj(x)
        return query, key, value
    
    def load_kv_cache(self,kv_cache,key,value):
        if kv_cache is not None:
            key = torch.concat([kv_cache.key,
                    key],dim=2) # B,H,S,E. dim becomes 2
            value = torch.concat([kv_cache.value,
                      value],dim=2)
        return key,value
    
    def calculate_unmasked_attention_logits(self,query,key):
        # Q.K' : [B,S,E] @ [B,E,S]
        key_t = key.transpose(1,2) # Transpose to [B,E,S] by exchanging dim 1 and 2
        
        # scaling factor
        scale = math.sqrt(self.embed_dim)

        # Calculate logits
        unmasked_attention_logits = (query@key_t)/scale

        return unmasked_attention_logits
    
    def apply_causal_mask(self,attention_logits,mask_value=None):
        # lower trianglular matrix with 1s
        B,Sq,Sk = attention_logits.shape 

        device = attention_logits.device
        causal_mask = torch.tril(torch.ones(B,Sk,Sk)).to(device)
        causal_mask = causal_mask[:,-Sq:,-Sk:] # Trim off, for kv_cache, no-op if kv_cache is None

        if mask_value is None:
            mask_value = torch.finfo(attention_logits.dtype).min

        # replace upper triangular with -inf or a very large negative number, causal masking!
        masked_attention_logits = attention_logits.masked_fill_(causal_mask == 0, mask_value) 
        return masked_attention_logits
    
    def apply_attention_mask(self,attention_mask,masked_attention_logits,mask_value=None):
        if mask_value is None:
            mask_value = torch.finfo(masked_attention_logits.dtype).min
        masked_attention_logits = masked_attention_logits.masked_fill_(attention_mask[:,None,:] == 0, mask_value)
        return masked_attention_logits
    
    def calculate_attention_weights(self,masked_attention_logits):
        attention_weights = torch.softmax(masked_attention_logits, dim=-1)
        return attention_weights
    
    def calculate_final_output(self,attention_weights,value):
        attention_output = attention_weights@value
        final_output = self.output_proj(attention_output)
        return final_output
    
    def forward(self, x,attention_mask=None,kv_cache=None):
        # Construct Q, K, V
        query, key, value = self.construct_query_key_value(x)

        # Load kv_cache, no-op if None
        key, value = self.load_kv_cache(kv_cache,key,value)

        # Calculate logits
        unmasked_attention_logits = self.calculate_unmasked_attention_logits(query,key)

        # Apply causal masking
        masked_attention_logits = self.apply_causal_mask(unmasked_attention_logits)

        if attention_mask is not None:
            # Apply attention mask
            masked_attention_logits = self.apply_attention_mask(attention_mask,masked_attention_logits)

        # Calculate attention weights
        attention_weights = self.calculate_attention_weights(masked_attention_logits)

        # And finally, Calculate the final output
        attention_output = self.calculate_final_output(attention_weights,value)

        output = BaseOutput(attention_output=attention_output,
                            attention_weights=attention_weights,
                            key=key,
                            value=value)
        return output
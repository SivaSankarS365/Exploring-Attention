from imports import *
from utils.data_containers import BaseOutput, KVCache

class AttentionTestCase:
    def __init__(self,Module,B,S,E,model_kwargs):
        self.Module = Module
        self.B = B
        self.S = S
        self.E = E
        self.model_kwargs = model_kwargs

    def test_forward(self):
        z = torch.rand(self.B,self.S,self.E)

        module = self.Module(**self.model_kwargs)
        output = module(z)
        assert output.attention_output.shape == (self.B,self.S,self.E)
        print('Forward test passed!')

    def test_attention_mask(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left" # Important!,usually decoder only models require left padding

        
        texts = ["Something random","A bit longer text","something even longer than the two before!"]
        encoded_input = tokenizer(texts, return_tensors='pt',padding=True)
        attention_mask = encoded_input['attention_mask']
        B,S = attention_mask.shape

        z = torch.rand(B,S,self.E)

        module = self.Module(**self.model_kwargs)
        output = module(z,attention_mask)
        assert output.attention_output.shape == (B,S,self.E)

        print('Attention mask test passed!')

    def test_kv_cache(self):
        z = torch.rand(self.B,self.S,self.E)
        module = self.Module(**self.model_kwargs)

        output = module(z,attention_mask=None)
        cache = KVCache(output.key,output.value) # Construct kv cache

        # new token
        _z = torch.rand(self.B,1,self.E)

        # Without KV Caching
        z_new = torch.concat([z,_z],dim=1)
        output_new = module(z_new,attention_mask=None)

        # With KV Caching
        output_with_caching = module(_z,attention_mask=None,kv_cache=cache)

        assert (output_new.attention_output[:,-1:,:] - output_with_caching.attention_output).abs().max().item() < 1e-5,\
              "Attention output does not match: kv-caching"
        
        print('KV cache test passed!')
        
    def run(self):
        self.test_forward()
        self.test_attention_mask()
        self.test_kv_cache()
    

if __name__ == "__main__":
    from .self_attention import BaseSelfAttention
    AttentionTestCase(BaseSelfAttention,B=2,S=3,E=16,model_kwargs={'embed_dim':4}).run()
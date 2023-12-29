class KVCache:
    def __init__(self,key,value,**kwargs):
        self.key = key
        self.value = value
        for k,v in kwargs.items():
            setattr(self,k,v)


class BaseOutput:
    def __init__(self,**kwargs):
        for k,v in kwargs.items():
            setattr(self,k,v)

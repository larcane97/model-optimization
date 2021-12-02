import torch
import torch.nn as nn
import math
from src.modules.base_generator import GeneratorAbstract
from src.modules.ShuffleNetV2.network import ShuffleNetV1, ShuffleNetV2

class ShuffleNetV2Generator(GeneratorAbstract):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    @property
    def out_channel(self)->int:
        return self.args[0]
    
    def __call__(self,repeat:int=1):
        module = ShuffleNetV2(model_size='0.5x')
        print('>> pretrained loading..')
        weight = torch.load('/opt/ml/ShuffleNet-Series/ShuffleNetV2.0.5x.pth')
        new_weight = { '.'.join(k.split('.')[1:]) : v for k,v in weight['state_dict'].items() }
        msg = module.load_state_dict(new_weight)
        print(msg)
        
        
        return self._get_module(module)


class ShuffleNetV1Generator(GeneratorAbstract):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    @property
    def out_channel(self)->int:
        return self.args[0]
    
    def __call__(self,repeat:int=1):
        module = ShuffleNetV1(model_size='0.5x',group=8)
        print('>> pretrained loading..')
        weight = torch.load('/opt/ml/ShuffleNet-Series/snetv1_group8_0.5x.pth')
        new_weight = { '.'.join(k.split('.')[1:]) : v for k,v in weight['state_dict'].items() }
        msg = module.load_state_dict(new_weight)
        print(msg)
        
        
        return self._get_module(module)

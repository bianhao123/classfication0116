import timm
from pprint import pprint

import torch
model_names = timm.list_models('*coat*')

pprint(model_names)

model = timm.create_model('coat_mini')
data = torch.randn(1, 3, 224, 224)
output = model(data)
print(output.shape)

import torch
from hdr_project.model import SmallUNetHDR
m=SmallUNetHDR(base=32)
m.load_state_dict(torch.load('outputs/train_runs/default/best_model.pt', map_location='cpu'))
m.eval()
x=torch.full((1,3,256,256), 0.1)
with torch.no_grad():
    print('Model output mean per channel:', m(x).mean(dim=[0,2,3]))

import torch
import torch.nn as nn

class ClassicDNN(nn.Module):

    def __init__(self, num_feat, num_cls):
        super(ClassicDNN, self).__init__()
        self.hid1 = nn.Sequential(nn.Linear(num_feat, 64), nn.ReLU(),    nn.Dropout(0.4143619965361732) )
        self.hid2 = nn.Sequential(nn.Linear(64, 128),      nn.Sigmoid(), nn.Dropout(0.09225974322037533))
        self.hid3 = nn.Sequential(nn.Linear(128, 64),      nn.ReLU(),    nn.Dropout(0.20942239619394942))
        self.out  = nn.Sequential(nn.Linear(64, 1),  nn.Sigmoid())  # end with num_cls=2 layers if CrossEntropyLoss 

    def forward(self, x):
        x = self.hid1(x)
        x = self.hid2(x)
        x = self.hid3(x)
        x = self.out(x)
        # x = torch.softmax(x, dim=1)
        return x.view(-1)
        # return x  # return 2-dim structured score if CrossEntropyLoss 


def get_model(data_config, **kwargs):
    model = ClassicDNN(num_feat=len(data_config.input_dicts['highvars']), num_cls=len(data_config.label_value))
    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }
    return model, model_info


def get_loss(data_config, **kwargs):
    # return torch.nn.CrossEntropyLoss()
    return torch.nn.BCELoss()

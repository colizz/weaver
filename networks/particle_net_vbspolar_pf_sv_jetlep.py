import torch
from utils.nn.model.ParticleNet import ParticleNetTaggerVBSPolar_addJetLep


def weight_init(m):
    import torch.nn as nn
    if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')


def get_model(data_config, **kwargs):
    conv_params = [
        (8, (64, 64, 64)),
        (8, (96, 96, 96)),
        (8, (128, 128, 128)),
        ]
    fc_params = [(256, 0.1)]
    use_fusion = True

    pf_features_dims = len(data_config.input_dicts['pf1_features'])
    sv_features_dims = len(data_config.input_dicts['sv1_features'])
    jet_features_dims = len(data_config.input_dicts['jet_features'])
    lep_features_dims = len(data_config.input_dicts['lep_features'])
    num_classes = len(data_config.label_value)
    model = ParticleNetTaggerVBSPolar_addJetLep(pf_features_dims, 
                              sv_features_dims,
                              jet_features_dims,
                              lep_features_dims,
                              num_classes,
                              conv_params, fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=kwargs.get('use_fts_bn', False),
                              use_counts=kwargs.get('use_counts', True),
                              pf_input_dropout=kwargs.get('pf_input_dropout', None),
                              sv_input_dropout=kwargs.get('sv_input_dropout', None),
                              for_inference=kwargs.get('for_inference', False)
                              )
    model.apply(weight_init)

    model_info = {
        'input_names':list(data_config.input_names),
        'input_shapes':{k:((1,) + s[1:]) for k, s in data_config.input_shapes.items()},
        'output_names':['softmax'],
        'dynamic_axes':{**{k:{0:'N', 2:'n_' + k.split('_')[0]} for k in data_config.input_names}, **{'softmax':{0:'N'}}},
        }

    return model, model_info


def get_loss(data_config, **kwargs):
    return torch.nn.CrossEntropyLoss()
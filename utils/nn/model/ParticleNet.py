import numpy as np
import torch
import torch.nn as nn

'''Based on https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py.'''


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k + 1, dim=-1)[1][:, :, 1:]  # (batch_size, num_points, k)
    return idx


# v1 is faster on GPU
def get_graph_feature_v1(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(2, 1).reshape(-1, num_dims)  # -> (batch_size, num_points, num_dims) -> (batch_size*num_points, num_dims)
    fts = fts[idx, :].view(batch_size, num_points, k, num_dims)  # neighbors: -> (batch_size*num_points*k, num_dims) -> ...
    fts = fts.permute(0, 3, 1, 2)  # (batch_size, num_dims, num_points, k)
    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)
    return fts


# v2 is faster on CPU
def get_graph_feature_v2(x, k, idx):
    batch_size, num_dims, num_points = x.size()

    idx_base = torch.arange(0, batch_size, device=x.device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    fts = x.transpose(0, 1).reshape(num_dims, -1)  # -> (num_dims, batch_size, num_points) -> (num_dims, batch_size*num_points)
    fts = fts[:, idx].view(num_dims, batch_size, num_points, k)  # neighbors: -> (num_dims, batch_size*num_points*k) -> ...
    fts = fts.transpose(1, 0)  # (batch_size, num_dims, num_points, k)

    x = x.view(batch_size, num_dims, num_points, 1).repeat(1, 1, 1, k)
    fts = torch.cat((x, fts - x), dim=1)  # ->(batch_size, 2*num_dims, num_points, k)

    return fts


class EdgeConvBlock(nn.Module):
    r"""EdgeConv layer.
    Introduced in "`Dynamic Graph CNN for Learning on Point Clouds
    <https://arxiv.org/pdf/1801.07829>`__".  Can be described as follows:
    .. math::
       x_i^{(l+1)} = \max_{j \in \mathcal{N}(i)} \mathrm{ReLU}(
       \Theta \cdot (x_j^{(l)} - x_i^{(l)}) + \Phi \cdot x_i^{(l)})
    where :math:`\mathcal{N}(i)` is the neighbor of :math:`i`.
    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    batch_norm : bool
        Whether to include batch normalization on messages.
    """

    def __init__(self, k, in_feat, out_feats, batch_norm=True, activation=True, cpu_mode=False):
        super(EdgeConvBlock, self).__init__()
        self.k = k
        self.batch_norm = batch_norm
        self.activation = activation
        self.num_layers = len(out_feats)
        self.get_graph_feature = get_graph_feature_v2 if cpu_mode else get_graph_feature_v1

        self.convs = nn.ModuleList()
        for i in range(self.num_layers):
            self.convs.append(nn.Conv2d(2 * in_feat if i == 0 else out_feats[i - 1], out_feats[i], kernel_size=1, bias=False if self.batch_norm else True))

        if batch_norm:
            self.bns = nn.ModuleList()
            for i in range(self.num_layers):
                self.bns.append(nn.BatchNorm2d(out_feats[i]))

        if activation:
            self.acts = nn.ModuleList()
            for i in range(self.num_layers):
                self.acts.append(nn.ReLU())

        if in_feat == out_feats[-1]:
            self.sc = None
        else:
            self.sc = nn.Conv1d(in_feat, out_feats[-1], kernel_size=1, bias=False)
            self.sc_bn = nn.BatchNorm1d(out_feats[-1])

        if activation:
            self.sc_act = nn.ReLU()

    def forward(self, points, features):

        topk_indices = knn(points, self.k)
        x = self.get_graph_feature(features, self.k, topk_indices)

        for conv, bn, act in zip(self.convs, self.bns, self.acts):
            x = conv(x)  # (N, C', P, K)
            if bn:
                x = bn(x)
            if act:
                x = act(x)

        fts = x.mean(dim=-1)  # (N, C, P)

        # shortcut
        if self.sc:
            sc = self.sc(features)  # (N, C_out, P)
            sc = self.sc_bn(sc)
        else:
            sc = features

        return self.sc_act(sc + fts)  # (N, C_out, P)


class ParticleNet(nn.Module):

    def __init__(self,
                 input_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 for_segmentation=False,
                 return_before_fusion=False,
                 **kwargs):
        super(ParticleNet, self).__init__(**kwargs)

        self.use_fts_bn = use_fts_bn
        if self.use_fts_bn:
            self.bn_fts = nn.BatchNorm1d(input_dims)

        self.use_counts = use_counts

        self.edge_convs = nn.ModuleList()
        for idx, layer_param in enumerate(conv_params):
            k, channels = layer_param
            in_feat = input_dims if idx == 0 else conv_params[idx - 1][1][-1]
            self.edge_convs.append(EdgeConvBlock(k=k, in_feat=in_feat, out_feats=channels, cpu_mode=for_inference))

        self.use_fusion = use_fusion
        self.return_before_fusion = return_before_fusion
        if self.use_fusion and not self.return_before_fusion:
            in_chn = sum(x[-1] for _, x in conv_params)
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.for_segmentation = for_segmentation

        if not self.return_before_fusion:
            fcs = []
            for idx, layer_param in enumerate(fc_params):
                channels, drop_rate = layer_param
                if idx == 0:
                    in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
                else:
                    in_chn = fc_params[idx - 1][0]
                if self.for_segmentation:
                    fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                             nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
                else:
                    fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
            if self.for_segmentation:
                fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
            else:
                fcs.append(nn.Linear(fc_params[-1][0], num_classes))
            self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference

    def forward(self, points, features, mask=None):
#         print('points:\n', points)
#         print('features:\n', features)
        if mask is None:
            mask = (features.abs().sum(dim=1, keepdim=True) != 0)  # (N, 1, P)
        points *= mask
        features *= mask
        coord_shift = (mask == 0) * 1e9
        if self.use_counts:
            counts = mask.float().sum(dim=-1)
            counts = torch.max(counts, torch.ones_like(counts))  # >=1

        if self.use_fts_bn:
            fts = self.bn_fts(features) * mask
        else:
            fts = features
        outputs = []
        for idx, conv in enumerate(self.edge_convs):
            pts = (points if idx == 0 else fts) + coord_shift
            fts = conv(pts, fts) * mask
            if self.use_fusion:
                outputs.append(fts)
        if self.return_before_fusion:
            assert self.use_fusion == True
            return outputs

        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs, dim=1)) * mask

#         assert(((fts.abs().sum(dim=1, keepdim=True) != 0).float() - mask.float()).abs().sum().item() == 0)
        
        if self.for_segmentation:
            x = fts
        else:
            if self.use_counts:
                x = fts.sum(dim=-1) / counts  # divide by the real counts
            else:
                x = fts.mean(dim=-1)

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output


class FeatureConv(nn.Module):

    def __init__(self, in_chn, out_chn, **kwargs):
        super(FeatureConv, self).__init__(**kwargs)
        self.conv = nn.Sequential(
            nn.BatchNorm1d(in_chn),
            nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_chn),
            nn.ReLU()
            )

    def forward(self, x):
        return self.conv(x)


class ParticleNetTagger(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTagger, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, pf_points, pf_features, pf_mask, sv_points, sv_features, sv_mask):
        if self.pf_input_dropout:
            pf_mask = (self.pf_input_dropout(pf_mask) != 0).float()
            pf_points *= pf_mask
            pf_features *= pf_mask
        if self.sv_input_dropout:
            sv_mask = (self.sv_input_dropout(sv_mask) != 0).float()
            sv_points *= sv_mask
            sv_features *= sv_mask

        points = torch.cat((pf_points, sv_points), dim=2)
        features = torch.cat((self.pf_conv(pf_features * pf_mask) * pf_mask, self.sv_conv(sv_features * sv_mask) * sv_mask), dim=2)
        mask = torch.cat((pf_mask, sv_mask), dim=2)
        return self.pn(points, features, mask)


class ParticleNetTaggerVBSPolar(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTaggerVBSPolar, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, pf1_points, pf1_features, pf1_mask, pf2_points, pf2_features, pf2_mask, pf3_points, pf3_features, pf3_mask, sv1_points, sv1_features, sv1_mask, sv2_points, sv2_features, sv2_mask, sv3_points, sv3_features, sv3_mask):
        if self.pf_input_dropout:
            pf1_mask = (self.pf_input_dropout(pf1_mask) != 0).float()
            pf1_points *= pf1_mask
            pf1_features *= pf1_mask
            pf2_mask = (self.pf_input_dropout(pf2_mask) != 0).float()
            pf2_points *= pf2_mask
            pf2_features *= pf2_mask
            pf3_mask = (self.pf_input_dropout(pf3_mask) != 0).float()
            pf3_points *= pf3_mask
            pf3_features *= pf3_mask
        if self.sv_input_dropout:
            sv1_mask = (self.sv_input_dropout(sv1_mask) != 0).float()
            sv1_points *= sv1_mask
            sv1_features *= sv1_mask
            sv2_mask = (self.sv_input_dropout(sv2_mask) != 0).float()
            sv2_points *= sv2_mask
            sv2_features *= sv2_mask
            sv3_mask = (self.sv_input_dropout(sv3_mask) != 0).float()
            sv3_points *= sv3_mask
            sv3_features *= sv3_mask

        points = torch.cat((pf1_points, pf2_points, pf3_points, sv1_points, sv2_points, sv3_points), dim=2)
        features = torch.cat((self.pf_conv(pf1_features * pf1_mask) * pf1_mask, self.pf_conv(pf2_features * pf2_mask) * pf2_mask, self.pf_conv(pf3_features * pf3_mask) * pf3_mask, self.sv_conv(sv1_features * sv1_mask) * sv1_mask, self.sv_conv(sv2_features * sv2_mask) * sv2_mask, self.sv_conv(sv3_features * sv3_mask) * sv3_mask), dim=2)
        mask = torch.cat((pf1_mask, pf2_mask, pf3_mask, sv1_mask, sv2_mask, sv3_mask), dim=2)
        return self.pn(points, features, mask)

class ParticleNetTaggerVBSPolar_dummy(nn.Module):

    def __init__(self,
                 dummy_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTaggerVBSPolar_dummy, self).__init__(**kwargs)
        self.dm_conv = FeatureConv(dummy_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, dm_points, dm_features, dm_mask):
        features = self.dm_conv(dm_features * dm_mask) * dm_mask
        return self.pn(dm_points, features, dm_mask)

class ParticleNetTaggerVBSPolar_addJetLep(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 jet_features_dims,
                 lep_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTaggerVBSPolar_addJetLep, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        self.jet_conv = FeatureConv(jet_features_dims, 32)
        self.lep_conv = FeatureConv(lep_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, pf1_points, pf1_features, pf1_mask, pf2_points, pf2_features, pf2_mask, sv1_points, sv1_features, sv1_mask, sv2_points, sv2_features, sv2_mask, jet_points, jet_features, jet_mask, lep_points, lep_features, lep_mask):
        if self.pf_input_dropout:
            pf1_mask = (self.pf_input_dropout(pf1_mask) != 0).float()
            pf1_points *= pf1_mask
            pf1_features *= pf1_mask
            pf2_mask = (self.pf_input_dropout(pf2_mask) != 0).float()
            pf2_points *= pf2_mask
            pf2_features *= pf2_mask
        if self.sv_input_dropout:
            sv1_mask = (self.sv_input_dropout(sv1_mask) != 0).float()
            sv1_points *= sv1_mask
            sv1_features *= sv1_mask
            sv2_mask = (self.sv_input_dropout(sv2_mask) != 0).float()
            sv2_points *= sv2_mask
            sv2_features *= sv2_mask

        points = torch.cat((pf1_points, pf2_points, sv1_points, sv2_points, jet_points, lep_points), dim=2)
        features = torch.cat((self.pf_conv(pf1_features * pf1_mask) * pf1_mask, self.pf_conv(pf2_features * pf2_mask) * pf2_mask, self.sv_conv(sv1_features * sv1_mask) * sv1_mask, self.sv_conv(sv2_features * sv2_mask) * sv2_mask, self.jet_conv(jet_features), self.lep_conv(lep_features)), dim=2)
        mask = torch.cat((pf1_mask, pf2_mask, sv1_mask, sv2_mask, jet_mask, lep_mask), dim=2)
        return self.pn(points, features, mask)

class ParticleNetTaggerVBSPolar_addLepMETAtBegin(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 lep_features_dims,
                 met_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 **kwargs):
        super(ParticleNetTaggerVBSPolar_addLepMETAtBegin, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv = FeatureConv(pf_features_dims, 32)
        self.sv_conv = FeatureConv(sv_features_dims, 32)
        self.lep_conv = FeatureConv(lep_features_dims, 32)
        self.met_conv = FeatureConv(met_features_dims, 32)
        self.pn = ParticleNet(input_dims=32,
                              num_classes=num_classes,
                              conv_params=conv_params,
                              fc_params=fc_params,
                              use_fusion=use_fusion,
                              use_fts_bn=use_fts_bn,
                              use_counts=use_counts,
                              for_inference=for_inference)

    def forward(self, pf1_points, pf1_features, pf1_mask, pf2_points, pf2_features, pf2_mask, sv1_points, sv1_features, sv1_mask, sv2_points, sv2_features, sv2_mask, lep_points, lep_features, lep_mask, met_points, met_features, met_mask):
        if self.pf_input_dropout:
            pf1_mask = (self.pf_input_dropout(pf1_mask) != 0).float()
            pf1_points *= pf1_mask
            pf1_features *= pf1_mask
            pf2_mask = (self.pf_input_dropout(pf2_mask) != 0).float()
            pf2_points *= pf2_mask
            pf2_features *= pf2_mask
        if self.sv_input_dropout:
            sv1_mask = (self.sv_input_dropout(sv1_mask) != 0).float()
            sv1_points *= sv1_mask
            sv1_features *= sv1_mask
            sv2_mask = (self.sv_input_dropout(sv2_mask) != 0).float()
            sv2_points *= sv2_mask
            sv2_features *= sv2_mask

        points = torch.cat((met_points, lep_points, pf1_points, pf2_points, sv1_points, sv2_points), dim=2)  # met & lep placed at the begining
        features = torch.cat((self.met_conv(met_features), self.lep_conv(lep_features), self.pf_conv(pf1_features * pf1_mask) * pf1_mask, self.pf_conv(pf2_features * pf2_mask) * pf2_mask, self.sv_conv(sv1_features * sv1_mask) * sv1_mask, self.sv_conv(sv2_features * sv2_mask) * sv2_mask), dim=2)
        mask = torch.cat((met_mask, lep_mask, pf1_mask, pf2_mask, sv1_mask, sv2_mask), dim=2)
        return self.pn(points, features, mask)

class ParticleNetTaggerVBSPolarV2(nn.Module):

    def __init__(self,
                 pf_features_dims,
                 sv_features_dims,
                 num_classes,
                 conv_params=[(7, (32, 32, 32)), (7, (64, 64, 64))],
                 fc_params=[(128, 0.1)],
                 use_fusion=True,
                 use_fts_bn=True,
                 use_counts=True,
                 pf_input_dropout=None,
                 sv_input_dropout=None,
                 for_inference=False,
                 for_segmentation=False,
                 **kwargs):
        super(ParticleNetTaggerVBSPolarV2, self).__init__(**kwargs)
        self.pf_input_dropout = nn.Dropout(pf_input_dropout) if pf_input_dropout else None
        self.sv_input_dropout = nn.Dropout(sv_input_dropout) if sv_input_dropout else None
        self.pf_conv1 = FeatureConv(pf_features_dims, 32)
        self.pf_conv2 = FeatureConv(pf_features_dims, 32)
        self.sv_conv1 = FeatureConv(sv_features_dims, 32)
        self.sv_conv2 = FeatureConv(sv_features_dims, 32)
        self.pn1 = ParticleNet(input_dims=32,
                               num_classes=num_classes,
                               conv_params=conv_params,
                               fc_params=fc_params,
                               use_fusion=use_fusion,
                               use_fts_bn=use_fts_bn,
                               use_counts=use_counts,
                               for_inference=for_inference,
                               return_before_fusion=True)
        self.pn2 = ParticleNet(input_dims=32,
                               num_classes=num_classes,
                               conv_params=conv_params,
                               fc_params=fc_params,
                               use_fusion=use_fusion,
                               use_fts_bn=use_fts_bn,
                               use_counts=use_counts,
                               for_inference=for_inference,
                               return_before_fusion=True)

        self.use_fusion = use_fusion
        assert self.use_fusion == True
        if self.use_fusion:
            in_chn = 2 * sum(x[-1] for _, x in conv_params)  # fusing two sets of output
            out_chn = np.clip((in_chn // 128) * 128, 128, 1024)
            self.fusion_block = nn.Sequential(nn.Conv1d(in_chn, out_chn, kernel_size=1, bias=False), nn.BatchNorm1d(out_chn), nn.ReLU())

        self.for_segmentation = for_segmentation
        fcs = []
        for idx, layer_param in enumerate(fc_params):
            channels, drop_rate = layer_param
            if idx == 0:
                in_chn = out_chn if self.use_fusion else conv_params[-1][1][-1]
            else:
                in_chn = fc_params[idx - 1][0]
            if self.for_segmentation:
                fcs.append(nn.Sequential(nn.Conv1d(in_chn, channels, kernel_size=1, bias=False),
                                         nn.BatchNorm1d(channels), nn.ReLU(), nn.Dropout(drop_rate)))
            else:
                fcs.append(nn.Sequential(nn.Linear(in_chn, channels), nn.ReLU(), nn.Dropout(drop_rate)))
        if self.for_segmentation:
            fcs.append(nn.Conv1d(fc_params[-1][0], num_classes, kernel_size=1))
        else:
            fcs.append(nn.Linear(fc_params[-1][0], num_classes))
        self.fc = nn.Sequential(*fcs)

        self.for_inference = for_inference
        self.use_counts = use_counts

    def forward(self, pf1_points, pf1_features, pf1_mask, pf2_points, pf2_features, pf2_mask, sv1_points, sv1_features, sv1_mask, sv2_points, sv2_features, sv2_mask):
        if self.pf_input_dropout:
            pf1_mask = (self.pf_input_dropout(pf1_mask) != 0).float()
            pf1_points *= pf1_mask
            pf1_features *= pf1_mask
            pf2_mask = (self.pf_input_dropout(pf2_mask) != 0).float()
            pf2_points *= pf2_mask
            pf2_features *= pf2_mask
        if self.sv_input_dropout:
            sv1_mask = (self.sv_input_dropout(sv1_mask) != 0).float()
            sv1_points *= sv1_mask
            sv1_features *= sv1_mask
            sv2_mask = (self.sv_input_dropout(sv2_mask) != 0).float()
            sv2_points *= sv2_mask
            sv2_features *= sv2_mask

        # print(pf1_points.shape, pf1_features.shape, pf1_mask.shape) # mask shape: (N, 1, P)
        points1 = torch.cat((pf1_points, sv1_points), dim=2)
        features1 = torch.cat((self.pf_conv1(pf1_features * pf1_mask) * pf1_mask, self.sv_conv1(sv1_features * sv1_mask) * sv1_mask), dim=2)
        mask1 = torch.cat((pf1_mask, sv1_mask), dim=2)
        points2 = torch.cat((pf2_points, sv2_points), dim=2)
        features2 = torch.cat((self.pf_conv2(pf2_features * pf2_mask) * pf2_mask, self.sv_conv2(sv2_features * sv2_mask) * sv2_mask), dim=2)
        mask2 = torch.cat((pf2_mask, sv2_mask), dim=2)

        outputs_pn1 = self.pn1(points1, features1, mask1)
        outputs_pn2 = self.pn2(points2, features2, mask2)

        # print(outputs_pn1[0].shape, outputs_pn1[1].shape, outputs_pn1[2].shape) # (N, 96, P)
        if self.use_fusion:
            fts = self.fusion_block(torch.cat(outputs_pn1 + outputs_pn2, dim=1))

        if self.for_segmentation:
            x = fts
        else:
            x = fts.mean(dim=-1)

        output = self.fc(x)
        if self.for_inference:
            output = torch.softmax(output, dim=1)
        # print('output:\n', output)
        return output
import math

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import copy
from typing import Optional, List

from networks.mh_att import MultiheadAttention

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 20 ** (2 * (dim_t // 2) / 128)  # as same as the pos in encoder
    # assert pos_tensor.size(-1) in [2, 4], "Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1))

    poses = []
    for i in range(pos_tensor.size(-1)):
        embed = pos_tensor[:, :, i] * scale
        pos = embed[:, :, None] / dim_t
        pos = torch.stack((pos[:, :, 0::2].sin(), pos[:, :, 1::2].cos()), dim=3).flatten(2)
        poses.append(pos)

    poses = torch.cat(poses, dim=2)
    return poses


class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()
        self.newconvinit = True
        self.d_model = d_model
        self.ref_point_head = MLP(2 // 2 * self.d_model, self.d_model, self.d_model, 2)

        self.nhead = nhead
        if self.newconvinit:
            self.point_transfer = nn.Sequential(
                nn.Conv3d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(32, d_model),
                nn.ReLU(),
                nn.Conv3d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(32, d_model),
                nn.ReLU(),
                nn.Conv3d(d_model, d_model, kernel_size=3, stride=1, padding=1, bias=True),
                nn.GroupNorm(32, d_model),
                nn.ReLU(),
            )
            for l in self.point_transfer.modules():
                if isinstance(l, nn.Conv3d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, refpoint_embed, refbbox_embed, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, d, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        refpoint_embed = refpoint_embed.unsqueeze(1).repeat(1, bs, 1)
        # print(refpoint_embed.shape)
        refbbox_embed = refbbox_embed.unsqueeze(1).repeat(1, bs, 1)
        mask = mask.flatten(1)
        # print('query_emb', query_embed.shape)
        # print('src shape', src.shape)
        # print('pos embed', pos_embed.shape)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        # print('memory', memory.shape)

        if self.newconvinit:
            memory_3d = memory.reshape(d, h, w, bs, c).permute(3, 4, 0, 1, 2)
            memory_3d = self.point_transfer(memory_3d)
            center_pos = torch.tensor([0.5, 0.5, 0.5]).cuda()
            # print(query_embed.sigmoid().transpose(0, 1).shape)
            tgt = F.grid_sample(memory_3d,
                                (refpoint_embed.sigmoid().transpose(0, 1).unsqueeze(1).unsqueeze(1) - center_pos[None, None, None, None,
                                                                                         :]) * 3,
                                mode="bilinear", padding_mode="zeros", align_corners=False)  # [bt, d_model, h, w]
            # print('tgt_after_sample', tgt.shape)
            tgt = tgt.flatten(2).permute(2, 0, 1)
            # print('tgt_after_flatten', tgt.shape)

        else:
            tgt = torch.zeros_like(refpoint_embed)
        # print('tgt', tgt.shape)

        hs, reference_points, reference_bboxes = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                                                              pos=pos_embed, refpoints_unsigmoid=refpoint_embed,
                                                              refbboxes_unsigmoid=refbbox_embed)

        return hs, reference_points, reference_bboxes
        # previous
        # hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
        #                   pos=pos_embed, query_pos=query_embed)
        # return hs.transpose(1, 2), memory.permute(1, 2, 0).view(bs, c, d, h, w)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output

def box_xyltrb_to_cxcywh(x, y):
    x_k, y_k, z_k = x.unbind(-1)
    l, t, r, b, f, bk = y.unbind(-1)
    box = [(x_k + 0.5 * (r-l)),
           (y_k + 0.5 * (b-t)),
           (z_k + 0.5 * (bk-f)),
           (l + r), (t + b), (f + bk)]
    return torch.stack(box, dim=-1)


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=True, query_scale_type='cond_elewise'):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        self.query_dim = 3
        self.bbox_dim = 6
        hidden_dim = 384
        self.ref_point_head = MLP(self.query_dim // 2 * hidden_dim, hidden_dim, hidden_dim, 2)
        self.ref_bbox_head = MLP(self.bbox_dim // 3 * hidden_dim, hidden_dim, hidden_dim, 2)
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        # self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(6)])
        # self.point_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        # self.point_embed = nn.ModuleList([self.point_embed for _ in range(6)])
        self.bbox_embed = None
        self.point_embed = None
        assert query_scale_type in ['cond_elewise', 'cond_scalar', 'fix_elewise']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(hidden_dim, hidden_dim, hidden_dim, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(hidden_dim, hidden_dim, 1, 2)
        elif query_scale_type == 'fix_elewise':
            self.query_scale = nn.Embedding(num_layers, hidden_dim)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))

    def update_point_or_bbox(self, tmp, reference, original=None):
        dim_ref = tmp.shape[-1]
        # print('dim_ref', dim_ref)
        assert dim_ref in [self.query_dim, self.bbox_dim]
        if dim_ref == self.bbox_dim:
            tmp[..., :dim_ref] += inverse_sigmoid(reference)
            new_reference = tmp[..., :dim_ref].sigmoid()
        if dim_ref == self.query_dim:
            # tmp[..., :dim_ref] += inverse_sigmoid(reference - original)
            # if original.shape[0] == 306:
            #     new_reference = tmp[..., :dim_ref].sigmoid() * torch.tensor([[1 / 18, 1 / 17]]).to(tmp.device)
            # else:
            #     new_reference = tmp[..., :dim_ref].sigmoid() * (1 / math.sqrt(original.shape[0]))
            # new_reference = new_reference + original
        #              # ablation for move the grid scale
            tmp[..., :dim_ref] += inverse_sigmoid(reference)
            new_reference = tmp[..., :dim_ref].sigmoid()
        return new_reference

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                refbboxes_unsigmoid: Optional[Tensor] = None,
                key_pos: Optional[Tensor] = None):

        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        reference_bboxes = refbboxes_unsigmoid.sigmoid()
        original_points = reference_points
        ref_points = [reference_points]
        ref_bboxes = [reference_bboxes]
        # print('reference_bboxes', reference_bboxes.shape)
        for layer_id, layer in enumerate(self.layers):
            # get sine embedding for the query vector
            obj_point = reference_points[..., :self.query_dim]  # [num_queries, batch_size, 3]
            obj_bbox = reference_bboxes[..., :self.bbox_dim]  # [num_queries, batch_size, 6]
            query_sine_embed = gen_sineembed_for_position(obj_point)  # [num_queries, batch_size, d_model]
            bbox_query_sine_embed = gen_sineembed_for_position(
                torch.cat([obj_point - obj_bbox[..., :3], obj_point + obj_bbox[..., 3:]],
                          dim=-1))  # [num_queries, batch_size, 2*d_model]
            query_pos = self.ref_point_head(query_sine_embed)  # [num_queries, batch_size, d_model]
            bbox_query_pos = self.ref_bbox_head(bbox_query_sine_embed)  # [num_queries, batch_size, d_model]

            # For the first decoder layer, we do not apply transformation over p_s
            if self.query_scale_type != 'fix_elewise':
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(output)

            else:
                pos_transformation = self.query_scale.weight[layer_id]

            #             apply transformation
            query_sine_embed = query_sine_embed * pos_transformation
            pos_sine_embed = pos

            # add box transformation
            if layer_id != 0:
                bbox_query_sine_embed = bbox_query_sine_embed * pos_transformation.repeat(1, 1, 2)

            # print('reference_points', reference_points.shape)
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos_sine_embed, query_pos=query_pos, query_sine_embed=query_sine_embed,
                           bbox_query_pos=bbox_query_pos, bbox_query_sine_embed=bbox_query_sine_embed,
                           is_first=(layer_id == 0), point_pos=reference_points,
                           bbox_ltrb=reference_bboxes)
            # iter update
            if self.bbox_embed is not None:
                tmp_bbox = self.bbox_embed[layer_id](output)
                tmp_point = self.point_embed[layer_id](output)

                new_reference_bboxes = self.update_point_or_bbox(tmp_bbox, reference_bboxes)
                new_reference_points = self.update_point_or_bbox(tmp_point, reference_points, original_points)

                if layer_id != self.num_layers - 1:
                    ref_bboxes.append(new_reference_bboxes)
                    # ref_points.append(reference_points)
                    ref_points.append(new_reference_points)

                # reference_bboxes = new_reference_bboxes.detach()
                # reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            if self.bbox_embed is not None:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    torch.stack(ref_points).transpose(1, 2),
                    torch.stack(ref_bboxes).transpose(1, 2),
                ]
            else:
                return [
                    torch.stack(intermediate).transpose(1, 2),
                    reference_points.unsqueeze(0).transpose(1, 2),
                    reference_bboxes.unsqueeze(0).transpose(1, 2),
                ]


        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        # print(tensor.shape)
        # print(pos.shape)
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False, keep_query_pos=False,
                 rm_self_attn_decoder=False, sdg=True):
        super().__init__()
        # Decoder Self-Attention
        if not rm_self_attn_decoder:
            self.sa_qcontent_proj = nn.Linear(d_model, d_model)
            self.sa_point_qpos_proj = nn.Linear(d_model, d_model)
            self.sa_bbox_qpos_proj = nn.Linear(d_model, d_model)

            self.sa_kcontent_proj = nn.Linear(d_model, d_model)
            self.sa_point_kpos_proj = nn.Linear(d_model, d_model)
            self.sa_bbox_kpos_proj = nn.Linear(d_model, d_model)

            self.sa_v_proj = nn.Linear(d_model, d_model)
            self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, vdim=d_model)

            self.norm1 = nn.LayerNorm(d_model)
            self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_point_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_bbox_qpos_proj = nn.Linear(d_model, d_model)

        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)

        self.ca_v_proj = nn.Linear(d_model, d_model)
        self.ca_bbox_qpos_sine_proj = nn.Linear(d_model * 2, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn = MultiheadAttention(d_model * 3, nhead, dropout=dropout, vdim=d_model)

        self.sdg = sdg
        if self.sdg:
            self.gaussian_proj = MLP(d_model, d_model, 6 * nhead, 3)  # if sdg is True

        self.nhead = nhead
        self.rm_self_attn_decoder = rm_self_attn_decoder

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before
        self.keep_query_pos = keep_query_pos

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    #     from visualizer import get_local
    #     @get_local('sa_attns', 'ca_attns')
    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None,
                query_sine_embed=None,
                bbox_query_pos: Optional[Tensor] = None,
                bbox_query_sine_embed=None,
                is_first=False,
                point_pos=None,
                bbox_ltrb=None,
                ):

        # ========== Begin of Self-Attention =============
        if not self.rm_self_attn_decoder:
            # Apply projections here
            # shape: num_queries x batch_size x 256
            # target is the input of the first decoder layer. zero by default.

            k_content = self.sa_kcontent_proj(tgt)
            k_point_pos = self.sa_point_kpos_proj(query_pos)
            k_bbox_pos = self.sa_bbox_kpos_proj(bbox_query_pos)
            v = self.sa_v_proj(tgt)

            q_content = self.sa_qcontent_proj(tgt)
            q_point_pos = self.sa_point_qpos_proj(query_pos)
            q_bbox_pos = self.sa_bbox_qpos_proj(bbox_query_pos)

            num_queries, bs, n_model = k_content.shape

            q = q_content + q_point_pos + q_bbox_pos
            k = k_content + k_point_pos + k_bbox_pos

            tgt2, attn_weights, attn_q, attn_k = self.self_attn(q, k, value=v, attn_mask=tgt_mask,
                                                                key_padding_mask=tgt_key_padding_mask)

            # only for visualize
            content_attn = torch.bmm(attn_q, attn_k.transpose(1, 2))
            sa_attns = content_attn

            # ========== End of Self-Attention =============

            tgt = tgt + self.dropout1(tgt2)
            tgt = self.norm1(tgt)

        bs, d, h, w = tgt.shape[1], 8, 8, 8
        mesh = torch.meshgrid(torch.arange(0, d), torch.arange(0, h), torch.arange(0, w))
        key_pos = torch.cat(
            [mesh[2].reshape(-1)[..., None], mesh[1].reshape(-1)[..., None], mesh[0].reshape(-1)[..., None]], -1).to(
            torch.device('cuda'))
        key_pos = key_pos.unsqueeze(0).repeat(bs, 1, 1)

        if self.sdg:
            # point_pos [num_queries, bs, 2]
            # key_pos [bs, len(memory), 2]
            num_queries = 12
            w, h, d = key_pos[..., 0].max().item() + 1, key_pos[..., 1].max().item() + 1, key_pos[..., 2].max().item() + 1
            memory_size = torch.tensor([w, h, d]).to(key_pos.device)
            # print(key_pos.shape)
            # print(point_pos.shape)
            key_pos = key_pos.repeat(self.nhead, 1, 1)

            gaussian_mapping = self.gaussian_proj(tgt)
            # print(gaussian_mapping.shape)
            offset = gaussian_mapping[..., :self.nhead * 3].tanh()

            point_pos = (point_pos * memory_size[None, None, :]).repeat(1, 1, self.nhead)
            bbox_ltrb = bbox_ltrb * memory_size[None, None, :].repeat(1, 1, 2)
            # print(bbox_ltrb.shape)
            bbox_ltrb = torch.stack((-bbox_ltrb[..., :3], bbox_ltrb[..., 3:]), dim=2).repeat(1, 1, 1, self.nhead)
            sample_offset = bbox_ltrb * offset.unsqueeze(2)
            sample_offset = sample_offset.max(-2)
            sample_offset = sample_offset[0] * (2 * sample_offset[1] - 1)
            sample_point_pos = point_pos + sample_offset
            # print(sample_point_pos.shape)
            sample_point_pos = sample_point_pos.reshape(num_queries, bs, self.nhead, 3).reshape(num_queries,
                                                                                                bs * self.nhead, 3)
            # print(gaussian_mapping.shape)
            # print('gaussian mapping', gaussian_mapping[..., self.nhead * 3:].shape)
            scale = gaussian_mapping[..., self.nhead * 3:].reshape(num_queries, bs, self.nhead, 3)\
                .reshape(num_queries, bs * self.nhead, 3)\
                .transpose(0, 1)

            relative_position = (key_pos + 0.5).unsqueeze(1) - sample_point_pos.transpose(0, 1).unsqueeze(2)
            gaussian_map = (relative_position.pow(2) * scale.unsqueeze(2).pow(2)).sum(-1)
            gaussian_map = -(gaussian_map - 0).abs() / 8.0

        else:
            gaussian_map = None

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt)
        k_content = self.ca_kcontent_proj(memory)
        v = self.ca_v_proj(memory)

        num_queries, bs, n_model = q_content.shape
        hw, _, _ = k_content.shape

        k_pos = self.ca_kpos_proj(pos)

        # For the first decoder layer, we add the positional embedding predicted from
        # the object query (the positional embedding) into the original query (key) in DETR.
        if is_first or self.keep_query_pos:
            # #             for transformation
            #             k_pos = self.ca_kpos_proj(pos)
            q_point_pos = self.ca_point_qpos_proj(query_pos)
            q_bbox_pos = self.ca_bbox_qpos_proj(bbox_query_pos)
            q = q_content + q_point_pos + q_bbox_pos
            k = k_content + k_pos
        else:
            q = q_content
            k = k_content

        # peca
        q = q.view(num_queries, bs, self.nhead, n_model // self.nhead)
        query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        query_sine_embed = query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)

        bbox_query_sine_embed = self.ca_bbox_qpos_sine_proj(bbox_query_sine_embed)
        bbox_query_sine_embed = bbox_query_sine_embed.view(num_queries, bs, self.nhead, n_model // self.nhead)
        q = torch.cat([q, query_sine_embed, bbox_query_sine_embed], dim=3).view(num_queries, bs, n_model * 3)

        k = k.view(hw, bs, self.nhead, n_model // self.nhead)
        k_pos = k_pos.view(hw, bs, self.nhead, n_model // self.nhead)
        k = torch.cat([k, k_pos, k_pos], dim=3).view(hw, bs, n_model * 3)

        # relative positional encoing as bias adding to the attention map before softmax operation
        tgt2, attn_weights, attn_q, attn_k = self.cross_attn(query=q, key=k,
                                                             value=v, attn_mask=memory_mask,
                                                             key_padding_mask=memory_key_padding_mask,
                                                             gaussian_map=gaussian_map)

        # ========== End of Cross-Attention =============

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list):
        x, mask = tensor_list
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        z_embed = not_mask.cumsum(3, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        # print(x_embed.shape)
        pos_x = x_embed[:, :, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, :, None] / dim_t
        pos_z = z_embed[:, :, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, :,0::2].sin(), pos_x[:, :, :, :,1::2].cos()), dim=5).flatten(4)
        pos_y = torch.stack((pos_y[:, :, :, :,0::2].sin(), pos_y[:, :, :, :,1::2].cos()), dim=5).flatten(4)
        pos_z = torch.stack((pos_z[:, :, :, :,0::2].sin(), pos_z[:, :, :, :,1::2].cos()), dim=5).flatten(4)
        # print(pos_z.shape)
        # print(pos_y.shape)
        # print(pos_x.shape)
        pos = torch.cat((pos_z, pos_y, pos_x), dim=4).permute(0, 4, 1, 2, 3)
        return pos


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, num_channels, transformer, num_classes=12, num_queries=12,
                 aux_loss=False, query_dim=3, bbox_dim=6,
                 iter_update=True,
                 bbox_embed_diff_each_layer=False,
                 class_embed_diff_each_layer=False,
                 first_independent_head=True,
                 ):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        num_decoder_layers = 6
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        # self.query_embed = nn.Embedding(num_queries, 3)  # .from_pretrained(init_pt_tensor).float()
        # self.query_embed = nn.Embedding(num_queries, hidden_dim).from_pretrained(pt_tensor).float()
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 6, 3)
        self.point_embed = MLP(hidden_dim, hidden_dim, 3, 3)
        self.class_embed = MLP(hidden_dim, hidden_dim, num_classes, 3)

        # setting query dim
        # when query_dim=2 refpoint_embed is the keypoint_embed
        self.query_dim = query_dim
        assert query_dim == 3
        # self.refpoint_embed = nn.Embedding(num_queries, query_dim).from_pretrained(init_pt_tensor).float()
        self.refpoint_embed = nn.Embedding(num_queries, query_dim)
        self.refpoint_embed.weight.data = self.refpoint_embed.weight.data/128
        # print(self.refpoint_embed.weight.data)
        # n = torch.arange(0, 1, 1 / math.pow(self.num_queries, 1/3))
        # mesh = torch.meshgrid(n, n, n)
        # reference_points = torch.cat([mesh[2].reshape(-1)[..., None], mesh[1].reshape(-1)[..., None], mesh[0].reshape(-1)[..., None]], -1)
        # print(self.refpoint_embed.weight.shape)
        # print(reference_points.shape)
        # self.refpoint_embed.weight.data[:, :3] = inverse_sigmoid(reference_points[:, :])
        self.refpoint_embed.weight.requires_grad = False  # learned or fixed ?

        # setting box dim
        # box_dim=4 box_dim is the left, top, right, and bottom referring the key point
        self.bbox_dim = bbox_dim
        assert bbox_dim == 6
        self.refbbox_embed = nn.Embedding(num_queries, bbox_dim)

        self.aux_loss = aux_loss
        self.iter_update = iter_update

        self.query_mlp = nn.Linear(hidden_dim, 3)
        self.input_proj = nn.Conv3d(num_channels, hidden_dim, kernel_size=1)
        self.aux_loss = aux_loss

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.point_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.point_embed.layers[-1].bias.data, 0)
        nn.init.constant_(self.class_embed.layers[-1].bias.data, bias_value)

        # share the prediction heads
        if bbox_embed_diff_each_layer:
            self.bbox_embed = _get_clones(self.bbox_embed, num_decoder_layers)
            self.point_embed = _get_clones(self.point_embed, num_decoder_layers)
        else:
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_decoder_layers)])
            self.point_embed = nn.ModuleList([self.point_embed for _ in range(num_decoder_layers)])

        if class_embed_diff_each_layer:
            self.class_embed = _get_clones(self.class_embed, num_decoder_layers)
        elif first_independent_head:
            independent_class_embed = copy.deepcopy(self.class_embed)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_decoder_layers)])
            self.class_embed[0] = independent_class_embed
        else:
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_decoder_layers)])

        if self.iter_update:
            self.transformer.decoder.bbox_embed = self.bbox_embed
            self.transformer.decoder.point_embed = self.point_embed

    # iter update
    def update_point_or_bbox(self, tmp, reference, original=None, reference_point=None):
        dim_ref = tmp.shape[-1]
        assert dim_ref in [self.query_dim, self.bbox_dim]
        if dim_ref == self.bbox_dim:
            tmp[..., :dim_ref] += inverse_sigmoid(reference)
            outputs = tmp[..., :dim_ref].sigmoid()
            outputs = box_xyltrb_to_cxcywh(reference_point, outputs)
        if dim_ref == self.query_dim:
            # tmp[..., :dim_ref] += inverse_sigmoid(reference - original)
            # if original.shape[1] == 306:
            #     new_reference = tmp[..., :dim_ref].sigmoid() * torch.tensor([[1 / 18, 1 / 17]]).to(tmp.device)
            # else:
            #     new_reference = tmp[..., :dim_ref].sigmoid() * torch.tensor([[1 / 20.0]]).to(
            #         tmp.device)  # (1 / math.sqrt(original.shape[1]))
            # # print(new_reference)
            # # print(original)
            # outputs = new_reference + original
        #              # ablation for move the grid scale
            tmp[..., :dim_ref] += inverse_sigmoid(reference)
            outputs = tmp[..., :dim_ref].sigmoid()
        return outputs

    def forward(self, input):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        features, pos, mask = input
        # print(mask.shape)
        src = features
        assert mask is not None
        refpoint_embedweight = self.refpoint_embed.weight
        refbbox_embedweight = self.refbbox_embed.weight
        hs, reference_point, reference_bbox = self.transformer(self.input_proj(src),
                                                               mask, refpoint_embedweight,
                                                               refbbox_embedweight, pos)
        outputs_coords = []
        outputs_classes = []
        outputs_objs = []
        outputs_points = []
        for lvl in range(hs.shape[0]):
            bbox_embed = self.bbox_embed[lvl]
            point_embed = self.point_embed[lvl]
            class_embed = self.class_embed[lvl]

            tmp_bbox = bbox_embed(hs[lvl])
            tmp_point = point_embed(hs[lvl])
            outputs_point = self.update_point_or_bbox(tmp_point, reference_point[lvl], original=reference_point[0])
            outputs_coord = self.update_point_or_bbox(tmp_bbox, reference_bbox[lvl], reference_point=outputs_point)
            outputs_class = class_embed(hs[lvl])
            # outputs_obj = self.obj_embed(hs[lvl])

            outputs_points.append(outputs_point)
            outputs_coords.append(outputs_coord)
            outputs_classes.append(outputs_class)
            # outputs_objs.append(outputs_obj)

        outputs_point = torch.stack(outputs_points)
        outputs_coord = torch.stack(outputs_coords)
        outputs_class = torch.stack(outputs_classes)
        # outputs_obj = torch.stack(outputs_objs)

        out = {'pred_logits': outputs_class[-1], 'pred_boxes': outputs_coord[-1]*128, 'pred_points': outputs_point[-1]*128}

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord, outputs_point)
        return out, 1


if __name__ == "__main__":
    hidden_dim = 384
    pos_enc = PositionEmbeddingSine(hidden_dim // 3, normalize=True).cuda()
    transformer = Transformer(d_model=hidden_dim, return_intermediate_dec=True).cuda()
    detr = DETR(hidden_dim, transformer, num_classes=12, num_queries=12).cuda()
    feature = torch.ones((4, 384, 8, 8, 8)).cuda()
    mask = torch.Tensor.bool(torch.zeros(4, 8, 8, 8)).cuda()
    pos_enc_f = pos_enc((feature, mask))
    out, query_emb = detr((feature, pos_enc_f, mask))
    # print(out['pred_logits'].shape)
    # print(out['pred_logits'][0][0])
    print(out['pred_boxes'].shape)






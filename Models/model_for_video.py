import torch
import torch.nn as nn
import torch.nn.functional as F
from fast_pytorch_kmeans import KMeans
from transformers import SegformerModel, SegformerConfig
from feature_aggregation_transformer import Feature_Aggregation_Transformer
from einops import rearrange


class NetworkInitializer:
    @staticmethod
    def init_weights(module):
        for child in module.children():
            if isinstance(child, nn.Conv2d):
                nn.init.kaiming_normal_(child.weight, mode='fan_in', nonlinearity='relu')
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if child.weight is not None:
                    nn.init.ones_(child.weight)
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, nn.Linear):
                nn.init.kaiming_normal_(child.weight, mode='fan_in', nonlinearity='relu')
                if child.bias is not None:
                    nn.init.zeros_(child.bias)
            elif isinstance(child, nn.Sequential):
                NetworkInitializer.init_weights(child)
            elif isinstance(child, (nn.ReLU, nn.ReLU6, nn.Upsample, nn.AdaptiveAvgPool2d, nn.Sigmoid)):
                continue
            else:
                if hasattr(child, 'initialize'):
                    child.initialize()


class PrototypeOperations:
    @staticmethod
    def cluster_features(features, num_clusters):
        B, _, H, W = features.size()
        masks = []
        for b in range(B):
            batch_feat = rearrange(features[b], 'c h w -> (h w) c')
            kmeans = KMeans(n_clusters=num_clusters, mode='euclidean', max_iter=10, verbose=0)
            cluster_labels = kmeans.fit_predict(batch_feat)
            cluster_masks = torch.zeros(num_clusters, H * W, device=features.device)
            for i in range(num_clusters):
                cluster_masks[i, cluster_labels == i] = 1
            cluster_masks = rearrange(cluster_masks, 'c (h w) -> c h w', h=H, w=W).unsqueeze(0)
            masks.append(cluster_masks)
        return torch.cat(masks, dim=0)

    @staticmethod
    def extract_prototypes_from_masks(features, masks, mode="softmax", dim=2):
        H, W = features.size(2), features.size(3)
        mask_probs = rearrange(masks, 'b c h w -> b c (h w)')
        if mode == "softmax":
            attention_weights = F.softmax(mask_probs, dim=dim)
        elif mode == "sigmoid":
            attention_weights = torch.sigmoid(mask_probs)
        else:
            attention_weights = mask_probs
        feat_flat = rearrange(features, 'b c h w -> b c (h w)')
        prototypes = torch.bmm(attention_weights, feat_flat.transpose(1, 2))
        vis_mask = rearrange(attention_weights, 'b c (h w) -> b c h w', h=H, w=W)
        return prototypes, vis_mask

    @staticmethod
    def extract_kmeans_prototypes(features, num_clusters):
        masks = PrototypeOperations.cluster_features(features, num_clusters)
        mask_probs = rearrange(masks, 'b c h w -> b c (h w)')
        feat_flat = rearrange(features, 'b c h w -> b c (h w)')
        prototypes = torch.bmm(mask_probs, feat_flat.transpose(1, 2))
        return prototypes

    @staticmethod
    def compute_similarity_knn(target_protos, ref_protos, k):
        target_norm = target_protos / (target_protos.norm(dim=2, keepdim=True) + 1e-8)
        ref_norm = ref_protos / (ref_protos.norm(dim=2, keepdim=True) + 1e-8)
        similarity = torch.matmul(target_norm, ref_norm.transpose(1, 2))
        topk_indices = torch.topk(similarity, k, dim=2)[1]
        return topk_indices

    @staticmethod
    def filter_prototypes(target_protos, ref_proto_list, k=16):
        B = target_protos.size(0)
        filtered_protos = None
        for ref_protos in ref_proto_list:
            topk_indices = PrototypeOperations.compute_similarity_knn(target_protos, ref_protos, k)
            filtered = []
            for b in range(B):
                filtered.append(ref_protos[b, topk_indices[b, :]])
            filtered_protos = torch.stack(filtered, dim=0)   
            filtered_protos = filtered_protos.permute(0, 2, 3, 1) 
        return filtered_protos

    @staticmethod
    def compute_correlation_map(features, prototypes):
        B, C, H, W = features.size()
        proto_norm = prototypes / (prototypes.norm(dim=2, keepdim=True) + 1e-8) 
        feat_flat = features.view(B, C, -1)
        feat_norm = feat_flat / (feat_flat.norm(dim=1, keepdim=True) + 1e-8)   
        correlation = torch.bmm(proto_norm, feat_norm).view(B, -1, H, W)   
        return correlation


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias=False)
        self.norm = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU(inplace=False)
    def forward(self, x):
        return self.activation(self.norm(self.conv(x)))
    def initialize(self):
        NetworkInitializer.init_weights(self)


class FeatureFusionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.activation = nn.ReLU(inplace=False)
        self.concat_conv = ConvBlock(channels * 2, channels, 3, padding=1)
    def forward(self, low_feat, high_feat):
        mult_feat = low_feat * high_feat
        concat_feat = torch.cat([low_feat, mult_feat], dim=1)
        output = self.concat_conv(concat_feat)
        return self.activation(output)
    def initialize(self):
        NetworkInitializer.init_weights(self)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.residual_path = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_ch, in_ch, 3, 1, 1))
        self.output_path = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False))
    def forward(self, x):
        residual = self.residual_path(x)
        x = F.relu(x + residual, inplace=False)
        return self.output_path(x)
    def initialize(self):
        NetworkInitializer.init_weights(self)


class AtrousFeatureModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.activation = nn.ReLU(False)
        self.bottleneck = nn.Sequential(ConvBlock(in_ch, out_ch, 1))
        self.atrous_6 = nn.Sequential(ConvBlock(out_ch, out_ch, 3, padding=6, dilation=6))
        self.atrous_12 = nn.Sequential(ConvBlock(out_ch, out_ch, 3, padding=12, dilation=12))
        self.atrous_18 = nn.Sequential(ConvBlock(out_ch, out_ch, 3, padding=18, dilation=18))
        self.channel_reduce = nn.Conv2d(in_ch, out_ch, 1)
    def forward(self, x):
        bottleneck_out = self.bottleneck(x)
        reduced = self.channel_reduce(x)
        atrous_1 = self.atrous_6(reduced)
        atrous_2 = self.atrous_12(reduced + atrous_1)
        atrous_3 = self.atrous_18(reduced + atrous_2)
        return self.activation(bottleneck_out + atrous_3)
    def initialize(self):
        NetworkInitializer.init_weights(self)

class SlotGenerator(nn.Module):
    def __init__(self, in_ch, out_ch, num_slots):
        super().__init__()
        self.channel_reducer = nn.Conv2d(in_ch, out_ch, 1)
        self.slot_encoder = nn.Conv2d(in_ch, num_slots, 1)
        self.feature_aggregator = Feature_Aggregation_Transformer(out_ch)
        self.proto_ops = PrototypeOperations()
    def forward(self, target_features, reference_features_list):
        reduced_target = self.channel_reducer(target_features)
        slot_logits = self.slot_encoder(target_features)
        target_protos = self.proto_ops.extract_kmeans_prototypes(reduced_target, 64)
        slot_protos, slot_masks = self.proto_ops.extract_prototypes_from_masks(reduced_target, slot_logits, "softmax", 1)
        ref_proto_list = []
        
        for ref_feat in reference_features_list:
            reduced_ref = self.channel_reducer(ref_feat)
            ref_proto, _ = self.proto_ops.extract_prototypes_from_masks(reduced_ref, reduced_ref, "softmax", 2)
            ref_proto_list.append(ref_proto)
        
        ref_proto_block = torch.stack(ref_proto_list, dim=1)               
        ref_proto_block = rearrange(ref_proto_block, 'b cn cl c -> b c cn cl')
        target_protos = rearrange(target_protos, 'b n c -> b c n')
        aggregated_protos = self.feature_aggregator(target_protos, ref_proto_block)
        aggregated_protos = rearrange(aggregated_protos, 'b c n -> b n c')
        ref_proto_list.append(aggregated_protos)
        return reduced_target, aggregated_protos, ref_proto_list, slot_protos, slot_masks
    def initialize(self):
        NetworkInitializer.init_weights(self)


class SlotAttentionModule(nn.Module):
    def __init__(self, channels, num_iterations=3):
        super().__init__()
        self.num_iterations = num_iterations
        self.feature_aggregator = Feature_Aggregation_Transformer(channels)
        self.proto_ops = PrototypeOperations()
    def forward(self, ref_proto_list, slot_protos):
        refined_slots = slot_protos
        for _ in range(self.num_iterations):
            filtered_refs = self.proto_ops.filter_prototypes(refined_slots, ref_proto_list)  
            filtered_refs = rearrange(filtered_refs, 'b cl c cn -> b c cn cl')             
            refined_slots = rearrange(refined_slots, 'b n c -> b c n')                   
            refined_slots = self.feature_aggregator(refined_slots, filtered_refs)          
            refined_slots = rearrange(refined_slots, 'b c n -> b n c')                     
        return refined_slots
    def initialize(self):
        NetworkInitializer.init_weights(self)


class VideoSegmentationNetwork(nn.Module):
    def __init__(self, num_slots=2, backbone_name="nvidia/mit-b2", feature_dim=128):
        super().__init__()

        cfg = SegformerConfig.from_pretrained(backbone_name)
        cfg.output_hidden_states = True
        self.rgb_backbone = SegformerModel.from_pretrained(backbone_name, config=cfg)
        self.flow_backbone = SegformerModel.from_pretrained(backbone_name, config=cfg)

        self.rgb_processor_s1 = AtrousFeatureModule(512, feature_dim)
        self.rgb_processor_s2 = AtrousFeatureModule(320, feature_dim)
        self.rgb_processor_s3 = AtrousFeatureModule(128, feature_dim)
        self.rgb_processor_s4 = AtrousFeatureModule(64, feature_dim)

        self.flow_processor_s1 = AtrousFeatureModule(512, feature_dim)
        self.flow_processor_s2 = AtrousFeatureModule(320, feature_dim)
        self.flow_processor_s3 = AtrousFeatureModule(128, feature_dim)
        self.flow_processor_s4 = AtrousFeatureModule(64, feature_dim)

        self.rgb_slot_gen = SlotGenerator(512, feature_dim, num_slots)
        self.flow_slot_gen = SlotGenerator(512, feature_dim, num_slots)
        self.rgb_slot_attn = SlotAttentionModule(feature_dim)
        self.flow_slot_attn = SlotAttentionModule(feature_dim)

        self.fuser_1 = FeatureFusionBlock(feature_dim)
        self.fuser_2 = FeatureFusionBlock(feature_dim)
        self.fuser_3 = FeatureFusionBlock(feature_dim)

        self.rgb_stream_fuser = nn.Conv2d(feature_dim + 48, feature_dim, 1)
        self.flow_stream_fuser = nn.Conv2d(feature_dim + 48, feature_dim, 1)

        self.cross_fuser_1 = ResidualBlock(feature_dim * 2, feature_dim)
        self.cross_fuser_2 = ResidualBlock(feature_dim * 2, feature_dim)
        self.cross_fuser_3 = ResidualBlock(feature_dim * 2, feature_dim)
        self.cross_fuser_4 = ResidualBlock(feature_dim * 2, feature_dim)

        self.pred_head_1 = nn.Conv2d(feature_dim, 1, 3, 1, 1)
        self.pred_head_2 = nn.Conv2d(feature_dim, 1, 3, 1, 1)
        self.pred_head_3 = nn.Conv2d(feature_dim, 1, 3, 1, 1)
        self.pred_head_4 = nn.Conv2d(feature_dim, 1, 3, 1, 1)

        self.proto_ops = PrototypeOperations()
        self.initialize()

    def extract_reference_features(self, rgb_refs, flow_refs):
        num_refs = rgb_refs.size(1) // 3
        rgb_ref_feats, flow_ref_feats = [], []
        with torch.no_grad():
            for n in range(num_refs):
                rgb_out = self.rgb_backbone(rgb_refs[:, n * 3:(n + 1) * 3, :, :])
                flow_out = self.flow_backbone(flow_refs[:, n * 3:(n + 1) * 3, :, :])
                rgb_ref_feats.append(rgb_out.hidden_states[-1])  
                flow_ref_feats.append(flow_out.hidden_states[-1])
        return rgb_ref_feats, flow_ref_feats

    def forward(self, rgb_input, flow_input, rgb_refs, flow_refs):
        orig_h, orig_w = rgb_input.shape[2:]

        rgb_ref_feats, flow_ref_feats = self.extract_reference_features(rgb_refs, flow_refs)

        rgb_feats = self.rgb_backbone(rgb_input).hidden_states
        flow_feats = self.flow_backbone(flow_input).hidden_states

        rgb_s4, rgb_s3, rgb_s2, rgb_s1 = rgb_feats[-1], rgb_feats[-2], rgb_feats[-3], rgb_feats[-4]
        flow_s4, flow_s3, flow_s2, flow_s1 = flow_feats[-1], flow_feats[-2], flow_feats[-3], flow_feats[-4]

        rgb_s1 = self.rgb_processor_s4(rgb_s1)
        rgb_s2 = self.rgb_processor_s3(rgb_s2)
        rgb_s3 = self.rgb_processor_s2(rgb_s3)
        rgb_s4 = self.rgb_processor_s1(rgb_s4)

        flow_s1 = self.flow_processor_s4(flow_s1)
        flow_s2 = self.flow_processor_s3(flow_s2)
        flow_s3 = self.flow_processor_s2(flow_s3)
        flow_s4 = self.flow_processor_s1(flow_s4)

        (rgb_reduced, rgb_agg_protos, rgb_ref_proto_list, rgb_slot_protos, rgb_slot_masks) = self.rgb_slot_gen(rgb_feats[-1], rgb_ref_feats)

        (flow_reduced, flow_agg_protos, flow_ref_proto_list, flow_slot_protos, flow_slot_masks) = self.flow_slot_gen(flow_feats[-1], flow_ref_feats)

        def safe_corr(tensor_a, tensor_b):
            try:
                return self.proto_ops.compute_correlation_map(tensor_a, tensor_b)
            except Exception as e:
                print(f"[Warning] correlation failed: {e}")
                return torch.zeros((tensor_a.size(0), 1, tensor_a.size(2), tensor_a.size(3)),device=tensor_a.device)

        rgb_corr = safe_corr(rgb_reduced, rgb_agg_protos)
        flow_corr = safe_corr(flow_reduced, flow_agg_protos)
        rgb_slot_corr = safe_corr(rgb_reduced, rgb_slot_protos)
        flow_slot_corr = safe_corr(flow_reduced, flow_slot_protos)

        def _resize_like(x, ref):
            if x.shape[2:] != ref.shape[2:]:
                return F.interpolate(x, size=ref.shape[2:], mode='bilinear', align_corners=False)
            return x

        rgb_corr = _resize_like(rgb_corr, rgb_s3)
        rgb_slot_corr = _resize_like(rgb_slot_corr, rgb_s3)
        flow_corr = _resize_like(flow_corr, flow_s3)
        flow_slot_corr = _resize_like(flow_slot_corr, flow_s3)

        rgb_cat = torch.cat([rgb_s3, rgb_corr, rgb_slot_corr], dim=1)
        flow_cat = torch.cat([flow_s3, flow_corr, flow_slot_corr], dim=1)

        if rgb_cat.shape[1] != self.rgb_stream_fuser.in_channels:
          if False: print(f"[AutoFix] Rebuilding rgb_stream_fuser: in={rgb_cat.shape[1]}")
          with torch.no_grad():
            self.rgb_stream_fuser = nn.Conv2d(rgb_cat.shape[1], self.rgb_stream_fuser.out_channels, 1).to(rgb_cat.device)

        if flow_cat.shape[1] != self.flow_stream_fuser.in_channels:
          if False: print(f"[AutoFix] Rebuilding flow_stream_fuser: in={flow_cat.shape[1]}")
          with torch.no_grad():
            self.flow_stream_fuser = nn.Conv2d(flow_cat.shape[1], self.flow_stream_fuser.out_channels, 1).to(flow_cat.device)

        rgb_s3 = self.rgb_stream_fuser(rgb_cat)
        flow_s3 = self.flow_stream_fuser(flow_cat)

        Ns = rgb_slot_corr.size(1)
        rgb_corr_ns = rgb_corr[:, :Ns, :, :]
        flow_corr_ns = flow_corr[:, :Ns, :, :]

        rgb_cat2 = torch.cat([rgb_s3, rgb_corr_ns, rgb_slot_corr], dim=1)
        flow_cat2 = torch.cat([flow_s3, flow_corr_ns, flow_slot_corr], dim=1)

        if rgb_cat2.shape[1] != self.rgb_stream_fuser.in_channels:
          if False: print(f"[AutoFix] Rebuilding rgb_stream_fuser (2): in={rgb_cat2.shape[1]}")
          with torch.no_grad():
            self.rgb_stream_fuser = nn.Conv2d(rgb_cat2.shape[1], self.rgb_stream_fuser.out_channels, 1).to(rgb_cat2.device)

        if flow_cat2.shape[1] != self.flow_stream_fuser.in_channels:
          if False: print(f"[AutoFix] Rebuilding flow_stream_fuser (2): in={flow_cat2.shape[1]}")
          with torch.no_grad():
            self.flow_stream_fuser = nn.Conv2d(flow_cat2.shape[1], self.flow_stream_fuser.out_channels, 1).to(flow_cat2.device)

        rgb_s3 = self.rgb_stream_fuser(rgb_cat2)
        flow_s3 = self.flow_stream_fuser(flow_cat2)

        fused_s1 = self.cross_fuser_1(torch.cat([rgb_s1, flow_s1], dim=1))
        fused_s2 = self.cross_fuser_2(torch.cat([rgb_s2, flow_s2], dim=1))
        fused_s3 = self.cross_fuser_3(torch.cat([rgb_s3, flow_s3], dim=1))
        fused_s4 = self.cross_fuser_4(torch.cat([rgb_s4, flow_s4], dim=1))

        fused_s2_up = F.interpolate(fused_s2, scale_factor=2, mode='bilinear', align_corners=False)
        fused_s3_up = F.interpolate(fused_s3, scale_factor=4, mode='bilinear', align_corners=False)
        fused_s4_up = F.interpolate(fused_s4, scale_factor=8, mode='bilinear', align_corners=False)

        decoded_s4 = fused_s4_up
        decoded_s3 = self.fuser_3(fused_s3_up, decoded_s4)
        decoded_s2 = self.fuser_2(fused_s2_up, decoded_s3)
        decoded_s1 = self.fuser_1(fused_s1, decoded_s2)

        pred_4 = self.pred_head_4(decoded_s4)
        pred_3 = self.pred_head_3(decoded_s3) + pred_4
        pred_2 = self.pred_head_2(decoded_s2) + pred_3
        pred_1 = self.pred_head_1(decoded_s1) + pred_2

        preds = [pred_1, pred_2, pred_3, pred_4]
        preds = [torch.sigmoid(self._resize_to(p, (orig_h, orig_w))) for p in preds]

        rgb_slot_corr = (rgb_slot_corr + 1) / 2
        flow_slot_corr = (flow_slot_corr + 1) / 2

        fg_rgb = rgb_slot_corr[:, 0:1, :, :]
        fg_flow = flow_slot_corr[:, 0:1, :, :]
        bg_rgb = rgb_slot_corr[:, 1:2, :, :]
        bg_flow = flow_slot_corr[:, 1:2, :, :]

        coarse_slot_rgb = self._resize_to(rgb_slot_masks[:, 0:1, :, :], (orig_h, orig_w))
        coarse_slot_flow = self._resize_to(flow_slot_masks[:, 0:1, :, :], (orig_h, orig_w))

        fine_slot_rgb_fg = self._resize_to(fg_rgb, (orig_h, orig_w))
        fine_slot_rgb_bg = self._resize_to(1 - bg_rgb, (orig_h, orig_w))
        fine_slot_flow_fg = self._resize_to(fg_flow, (orig_h, orig_w))
        fine_slot_flow_bg = self._resize_to(1 - bg_flow, (orig_h, orig_w))

        coarse_slots = [coarse_slot_rgb, coarse_slot_flow]
        fine_slots = [fine_slot_rgb_fg, fine_slot_rgb_bg, fine_slot_flow_fg, fine_slot_flow_bg]

        all_outputs = preds + coarse_slots + fine_slots
        return all_outputs, fine_slots

    @staticmethod
    def _resize_to(tensor, target_size):
        return F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)

    def initialize(self):
        NetworkInitializer.init_weights(self)

GSANet = VideoSegmentationNetwork

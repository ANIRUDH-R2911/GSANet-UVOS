import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForImageClassification


class NetworkInitializer:
    @staticmethod
    def init_weights(module: nn.Module):
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


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=1, pad=0, dilation=1, use_bias=False):
        super(ConvBlock, self).__init__()
        self.conv_layer = nn.Conv2d(
            in_ch, out_ch, kernel_size=k_size, stride=stride, padding=pad, dilation=dilation, bias=use_bias)
        self.norm_layer = nn.BatchNorm2d(out_ch)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.norm_layer(x)
        x = self.activation(x) 
        return x

    def initialize(self):
        NetworkInitializer.init_weights(self)


class FeatureFusionBlock(nn.Module):
    def __init__(self, ch):
        super(FeatureFusionBlock, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.concat_conv = ConvBlock(ch * 2, ch, 3, pad=1)

    def forward(self, low_feat, high_feat):
        mult_feat = low_feat * high_feat
        concat_feat = torch.cat([low_feat, mult_feat], dim=1)
        output = self.concat_conv(concat_feat)
        output = self.activation(output)
        return output

    def initialize(self):
        NetworkInitializer.init_weights(self)


class AtrousFeatureModule(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(AtrousFeatureModule, self).__init__()
        self.activation = nn.ReLU(True)
        self.bottleneck = ConvBlock(in_ch, out_ch, 1)
        self.atrous_6 = ConvBlock(out_ch, out_ch, 3, stride=1, pad=6, dilation=6)
        self.atrous_12 = ConvBlock(out_ch, out_ch, 3, stride=1, pad=12, dilation=12)
        self.atrous_18 = ConvBlock(out_ch, out_ch, 3, stride=1, pad=18, dilation=18)
        self.channel_reduce = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        bottleneck_out = self.bottleneck(x)
        reduced = self.channel_reduce(x)
        atrous_out_1 = self.atrous_6(reduced)
        atrous_out_2 = self.atrous_12(reduced + atrous_out_1)
        atrous_out_3 = self.atrous_18(reduced + atrous_out_2)

        output = self.activation(bottleneck_out + atrous_out_3)
        return output

    def initialize(self):
        NetworkInitializer.init_weights(self)


class SaliencyNetwork(nn.Module):

    def __init__(self, backbone_name: str = "nvidia/mit-b2", feature_dim: int = 128):
        super(SaliencyNetwork, self).__init__()
        self.backbone = SegformerForImageClassification.from_pretrained(
            backbone_name, use_safetensors=True)
        try:
            self.backbone.segformer.encoder.gradient_checkpointing = True
        except Exception:
            pass

        self.scale_processor_1 = AtrousFeatureModule(512, feature_dim)
        self.scale_processor_2 = AtrousFeatureModule(320, feature_dim)
        self.scale_processor_3 = AtrousFeatureModule(128, feature_dim)
        self.scale_processor_4 = AtrousFeatureModule(64, feature_dim)

        self.fuser_1 = FeatureFusionBlock(feature_dim)
        self.fuser_2 = FeatureFusionBlock(feature_dim)
        self.fuser_3 = FeatureFusionBlock(feature_dim)

        self.pred_head_1 = nn.Conv2d(feature_dim, 1, kernel_size=3, stride=1, padding=1)
        self.pred_head_2 = nn.Conv2d(feature_dim, 1, kernel_size=3, stride=1, padding=1)
        self.pred_head_3 = nn.Conv2d(feature_dim, 1, kernel_size=3, stride=1, padding=1)
        self.pred_head_4 = nn.Conv2d(feature_dim, 1, kernel_size=3, stride=1, padding=1)

        self.initialize()

    def forward(self, img: torch.Tensor):
        orig_h, orig_w = img.shape[2:]
        outputs = self.backbone(img, output_hidden_states=True, return_dict=True)
        hidden_states = outputs.hidden_states 
        feat_s1_raw, feat_s2_raw, feat_s3_raw, feat_s4_raw = hidden_states[-4:]

        feat_s1 = self.scale_processor_4(feat_s1_raw)  
        feat_s2 = self.scale_processor_3(feat_s2_raw)  
        feat_s3 = self.scale_processor_2(feat_s3_raw)  
        feat_s4 = self.scale_processor_1(feat_s4_raw)  
        base_h, base_w = feat_s1.shape[2:]
        feat_s1_up = feat_s1
        feat_s2_up = F.interpolate(feat_s2, size=(base_h, base_w), mode='bilinear', align_corners=False)
        feat_s3_up = F.interpolate(feat_s3, size=(base_h, base_w), mode='bilinear', align_corners=False)
        feat_s4_up = F.interpolate(feat_s4, size=(base_h, base_w), mode='bilinear', align_corners=False)

        fused_4 = feat_s4_up
        fused_3 = self.fuser_3(feat_s3_up, fused_4)
        fused_2 = self.fuser_2(feat_s2_up, fused_3)
        fused_1 = self.fuser_1(feat_s1_up, fused_2)

        pred_map_4 = self.pred_head_4(fused_4)
        pred_map_3 = self.pred_head_3(fused_3) + pred_map_4
        pred_map_2 = self.pred_head_2(fused_2) + pred_map_3
        pred_map_1 = self.pred_head_1(fused_1) + pred_map_2

        final_pred_1 = torch.sigmoid(self._resize_to(pred_map_1, (orig_h, orig_w)))
        final_pred_2 = torch.sigmoid(self._resize_to(pred_map_2, (orig_h, orig_w)))
        final_pred_3 = torch.sigmoid(self._resize_to(pred_map_3, (orig_h, orig_w)))
        final_pred_4 = torch.sigmoid(self._resize_to(pred_map_4, (orig_h, orig_w)))

        predictions = [final_pred_1, final_pred_2, final_pred_3, final_pred_4]
        return predictions, None

    @staticmethod
    def _resize_to(tensor: torch.Tensor, target_size):
        return F.interpolate(tensor, size=target_size, mode='bilinear', align_corners=False)

    def initialize(self):
        NetworkInitializer.init_weights(self)

GSANet = SaliencyNetwork

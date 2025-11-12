import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange


class LayerNormWrapper(nn.Module):    
    def __init__(self, dim, function):
        super(LayerNormWrapper, self).__init__()
        self.normalization = nn.LayerNorm(dim)
        self.function = function

    def forward(self, x, **kwargs):
        normalized = self.normalization(x)
        return self.function(normalized, **kwargs)


class MLPBlock(nn.Module):    
    def __init__(self, input_dim, hidden_dim, dropout_rate=0.):
        super(MLPBlock, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.Dropout(dropout_rate))

    def forward(self, x):
        return self.network(x)


class MultiHeadSelfAttention(nn.Module):    
    def __init__(self, dim, num_heads=8, head_dim=64, dropout_rate=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale_factor = head_dim ** -0.5
        
        inner_dim = num_heads * head_dim
        needs_projection = not (num_heads == 1 and head_dim == dim)

        self.softmax = nn.Softmax(dim=-1)
        self.qkv_projection = nn.Linear(dim, inner_dim * 3, bias=False)

        self.output_projection = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout_rate)
        ) if needs_projection else nn.Identity()

    def forward(self, x):
        batch_size, seq_len = x.shape[:2]
        qkv = self.qkv_projection(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),qkv)
        
        attention_scores = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale_factor
        attention_weights = self.softmax(attention_scores)
        
        attended = einsum('b h i j, b h j d -> b h i d', attention_weights, v)
        attended = rearrange(attended, 'b h n d -> b n (h d)')
        
        return self.output_projection(attended)


class TransformerBlock(nn.Module):    
    def __init__(self, dim, depth=1, num_heads=2, head_dim=32, dropout_rate=0.3):
        super(TransformerBlock, self).__init__()
        feedforward_dim = dim * 2
        
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LayerNormWrapper(
                    dim, 
                    MultiHeadSelfAttention(dim, num_heads, head_dim, dropout_rate)),
                LayerNormWrapper(
                    dim, 
                    MLPBlock(dim, feedforward_dim, dropout_rate))]))

    def forward(self, x):
        for attention_layer, feedforward_layer in self.layers:
            x = attention_layer(x) + x
            x = feedforward_layer(x) + x
        return x


class GlobalToLocalAttention(nn.Module):    
    def __init__(self, dim, num_heads, channels, dropout_rate=0.):
        super(GlobalToLocalAttention, self).__init__()
        inner_dim = num_heads * channels
        self.num_heads = num_heads
        self.scale_factor = channels ** -0.5
        
        self.query_proj = nn.Linear(dim, inner_dim)
        self.key_proj = nn.Linear(dim, inner_dim)
        self.value_proj = nn.Linear(dim, inner_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout_rate))

    def forward(self, global_feat, local_context):
        batch_size, num_context, _ = local_context.shape
        batch_size, channels, num_tokens = global_feat.shape
        global_reshaped = global_feat.transpose(1, 2).unsqueeze(1)
        queries = self.query_proj(local_context).view(batch_size, self.num_heads, num_context, channels)
        keys = self.key_proj(global_reshaped)
        values = self.value_proj(global_reshaped)
        
        attention_scores = queries @ keys.transpose(2, 3) * self.scale_factor
        attention_weights = self.softmax(attention_scores)
        
        attended = attention_weights @ values
        attended = rearrange(attended, 'b h m c -> b m (h c)')
        output_context = local_context + self.output_proj(attended)
        value_output = values.squeeze(1).transpose(1, 2)
        
        return value_output, output_context


class LocalToGlobalAttention(nn.Module):    
    def __init__(self, dim, num_heads, channels, dropout_rate=0.):
        super(LocalToGlobalAttention, self).__init__()
        inner_dim = num_heads * channels
        self.num_heads = num_heads
        self.scale_factor = channels ** -0.5
        
        self.query_proj = nn.Linear(dim, inner_dim)
        self.key_proj = nn.Linear(dim, inner_dim)
        self.value_proj = nn.Linear(dim, inner_dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.output_proj = nn.Sequential(
            nn.Linear(inner_dim, channels),
            nn.Dropout(dropout_rate))

    def forward(self, global_feat, local_context):
        batch_size, num_context, _ = local_context.shape
        batch_size, channels, num_tokens = global_feat.shape
        queries = self.query_proj(global_feat.transpose(1, 2).unsqueeze(1))
        keys = self.key_proj(local_context).view(batch_size, self.num_heads, num_context, channels)
        values = self.value_proj(local_context).view(batch_size, self.num_heads, num_context, channels)
        attention_scores = queries @ keys.transpose(2, 3) * self.scale_factor
        attention_weights = self.softmax(attention_scores)
        attended = attention_weights @ values
        attended = rearrange(attended, 'b h l c -> b l (h c)')
        output = self.output_proj(attended)
        output = output.permute(0, 2, 1)
        return global_feat + output


class BidirectionalAggregationBlock(nn.Module):    
    def __init__(self, dim, num_heads, channels, dropout_rate=0.):
        super(BidirectionalAggregationBlock, self).__init__()
        self.global_to_local = GlobalToLocalAttention(dim, num_heads, channels, dropout_rate)
        self.local_to_global = LocalToGlobalAttention(dim, num_heads, channels, dropout_rate)
        self.context_transformer = TransformerBlock(dim)
        self.feature_transformer = TransformerBlock(dim)

    def forward(self, global_features, local_features):
        local_context = local_features.permute(0, 2, 3, 1)
        local_context = rearrange(local_context, 'b cn cl c -> b (cn cl) c')
        intermediate_feat, updated_context = self.global_to_local(global_features, local_context)
        refined_context = self.context_transformer(updated_context)
        updated_global = self.local_to_global(intermediate_feat, refined_context)
        aggregated = self.feature_transformer(updated_global.permute(0, 2, 1))
        aggregated = aggregated.permute(0, 2, 1)
        return aggregated


class FeatureAggregationTransformer(nn.Module):
    def __init__(self, num_channels):
        super(FeatureAggregationTransformer, self).__init__()
        self.num_channels = num_channels
        intermediate_channels = num_channels
        self.local_proj = nn.Conv1d(num_channels, intermediate_channels, kernel_size=1, bias=False)
        
        self.aggregation_block = BidirectionalAggregationBlock(
            intermediate_channels, 
            num_heads=1, 
            channels=intermediate_channels)
        
        self.feature_norm = nn.Conv1d(intermediate_channels, intermediate_channels, kernel_size=1, bias=False)
        self.feature_refine = nn.Conv1d(intermediate_channels, intermediate_channels, kernel_size=1, bias=False)
     
        self.fusion = nn.Sequential(
            nn.Conv1d(intermediate_channels * 2, num_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(num_channels))
        
        self.spatial_attention = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)

    def forward(self, global_prototypes, local_references):
        spatial_weights = self.spatial_attention(local_references)
        spatial_weights = F.softmax(spatial_weights, dim=-1)
        weighted_local = local_references * spatial_weights  
        aggregated_local = torch.sum(weighted_local, dim=-1)  
        
        projected_local = self.local_proj(aggregated_local)
        aggregated_global = self.aggregation_block(global_prototypes, weighted_local)
        output = F.leaky_relu(aggregated_global, negative_slope=0.2)       
        return output

Feature_Aggregation_Transformer = FeatureAggregationTransformer
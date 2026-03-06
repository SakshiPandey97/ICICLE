import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.register_buffer("positions", torch.arange(0, max_len).long(), persistent=False)
    def forward(self, x):
        L = x.size(1)
        return x + self.pos_emb(self.positions[:L].unsqueeze(0))

class ALiBi(nn.Module):
    """Symmetric ALiBi attention bias: -|i-j| for bidirectional encoding."""
    def __init__(self, n_heads, max_len=2048):
        super().__init__()
        self.n_heads = n_heads
        slopes = torch.tensor([2 ** (-8 * (i + 1) / n_heads) for i in range(n_heads)])
        self.register_buffer("slopes", slopes.view(1, n_heads, 1, 1))
        
        positions = torch.arange(max_len)
        rel_pos = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        rel_pos = -rel_pos.abs()  
        self.register_buffer("rel_pos", rel_pos)
    
    def forward(self, seq_len):
        
        rel_pos = self.rel_pos[:seq_len, :seq_len]  
        alibi_bias = self.slopes * rel_pos.unsqueeze(0).unsqueeze(0)
        
        return alibi_bias

class TransformerEncoderLayerALiBi(nn.Module):
    """TransformerEncoderLayer with ALiBi bias support."""
    def __init__(self, d_model, n_heads, ffn_dim, dropout=0.1, use_alibi=True):
        super().__init__()
        self.use_alibi = use_alibi
        self.n_heads = n_heads
        self.d_model = d_model

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        self.linear1 = nn.Linear(d_model, ffn_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_dim, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
    
    def forward(self, x, src_key_padding_mask=None, attn_bias=None):
        
        x_norm = self.norm1(x)
        
        if self.use_alibi and attn_bias is not None:
            
            B, L, D = x_norm.shape
            
            qkv = torch.nn.functional.linear(x_norm, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias)
            qkv = qkv.reshape(B, L, 3, self.n_heads, D // self.n_heads)
            qkv = qkv.permute(2, 0, 3, 1, 4)  
            q, k, v = qkv[0], qkv[1], qkv[2]
            
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D // self.n_heads)  
            scores = scores + attn_bias  
            if src_key_padding_mask is not None:
                scores = scores.masked_fill(
                    src_key_padding_mask.unsqueeze(1).unsqueeze(2),  
                    float('-inf')
                )
            
            attn_weights = torch.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, v)  
            attn_output = attn_output.transpose(1, 2).reshape(B, L, D)  
            attn_output = torch.nn.functional.linear(attn_output, self.self_attn.out_proj.weight, self.self_attn.out_proj.bias)
        else:
            # Standard attention (no ALiBi)
            attn_output, _ = self.self_attn(
                x_norm, x_norm, x_norm,
                key_padding_mask=src_key_padding_mask
            )
        
        
        x = x + self.dropout1(attn_output)
        
        x_norm = self.norm2(x)
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x_norm))))
        x = x + self.dropout2(ff_output)
        
        return x

class AttentionPooling(nn.Module):
    """Learnable attention pooling"""
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Linear(d_model, 1)
    
    def forward(self, x, mask):
        attn_scores = self.attention(x).squeeze(-1)  
        attn_scores = attn_scores.masked_fill(mask, float('-inf'))
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1) 
        return pooled


class ConvStem(nn.Module):
    """Conv stem to capture local DNA motifs before transformer."""
    def __init__(self, d_model, dropout=0.1, kernel_sizes=[3, 5, 7]):
        super().__init__()
        
        self.kernel_sizes = kernel_sizes
        n_kernels = len(kernel_sizes)
        
        # Split channels across kernels, add remainder to first
        base_channels = d_model // n_kernels
        remainder = d_model - (base_channels * n_kernels)

        self.convs = nn.ModuleList()
        for i, k in enumerate(kernel_sizes):
            out_channels = base_channels + (remainder if i == 0 else 0)
            self.convs.append(
                nn.Conv1d(d_model, out_channels, kernel_size=k, padding=0)
            )
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()
    
    def forward(self, x):
        x_t = x.transpose(1, 2)
        
        conv_outputs = []
        for conv, k in zip(self.convs, self.kernel_sizes):
            pad_left = (k - 1) // 2
            pad_right = k - 1 - pad_left
            x_padded = F.pad(x_t, (pad_left, pad_right))
            conv_outputs.append(conv(x_padded))
        
        x_conv = torch.cat(conv_outputs, dim=1) 
        x_conv = x_conv.transpose(1, 2)
        
        x = x + self.dropout(self.activation(x_conv))
        x = self.norm(x)
        
        return x


class BERTForICE(nn.Module):
    def __init__(self, vocab_size, pad_id=1, max_len=512,
                 d_model=256, n_heads=4, n_layers=3, ffn_dim=1024,
                 dropout=0.1, num_labels=2, pooling='attention', use_alibi=False,
                 use_conv_stem=False):
        super().__init__()
        self.pad_id = pad_id
        self.pooling = pooling  
        self.use_alibi = use_alibi
        self.use_conv_stem = use_conv_stem
        self.n_heads = n_heads
        self.n_layers = n_layers
        
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
      
        if use_conv_stem:
            self.conv_stem = ConvStem(d_model, dropout=dropout)
        else:
            self.conv_stem = None
        
        if use_alibi:
            self.alibi = ALiBi(n_heads=n_heads, max_len=max_len)
            self.pos_emb = None
            self.encoder_layers = nn.ModuleList([
                TransformerEncoderLayerALiBi(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=ffn_dim,
                    dropout=dropout,
                    use_alibi=True
                )
                for _ in range(n_layers)
            ])
        else:
            self.pos_emb = PositionalEmbedding(max_len, d_model)
            self.alibi = None
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=ffn_dim,
                dropout=dropout, batch_first=True, activation="gelu", norm_first=True
            )
            self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
            self.encoder_layers = None
        
        self.drop = nn.Dropout(dropout)

        if self.pooling == 'attention':
            self.pooler = AttentionPooling(d_model)
        
        self.cls_head = nn.Linear(d_model, num_labels)

    def forward(self, input_ids):
        mask = (input_ids == self.pad_id)
        x = self.tok_emb(input_ids)

        if self.use_conv_stem and self.conv_stem is not None:
            x = self.conv_stem(x)

        if self.use_alibi:
            seq_len = input_ids.size(1)
            attn_bias = self.alibi(seq_len)

            for layer in self.encoder_layers:
                x = layer(x, src_key_padding_mask=mask, attn_bias=attn_bias)
        else:
            x = self.pos_emb(x)
            x = self.encoder(x, src_key_padding_mask=mask)

        if self.pooling == 'cls':
            pooled = x[:, 0, :]
        elif self.pooling == 'mean':
            mask_expanded = mask.unsqueeze(-1).expand_as(x)  
            x_masked = x.masked_fill(mask_expanded, 0.0)
            sum_x = x_masked.sum(dim=1)  
            seq_lengths = (~mask).sum(dim=1, keepdim=True).float()  
            pooled = sum_x / seq_lengths.clamp(min=1e-9)
        elif self.pooling == 'max':
            mask_expanded = mask.unsqueeze(-1).expand_as(x)
            x_masked = x.masked_fill(mask_expanded, float('-inf'))
            pooled = x_masked.max(dim=1)[0]  
        elif self.pooling == 'attention':
            pooled = self.pooler(x, mask)
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")
        
        return self.cls_head(self.drop(pooled))        

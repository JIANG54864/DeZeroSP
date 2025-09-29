import numpy as np
import dezerosp.layers as L
import dezerosp.functions as F
from dezerosp import Parameter, Variable
from dezerosp.layers import Layer

class MultiHeadAttention(Layer):
    """多头注意力机制"""
    def __init__(self, d_model, n_head, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head
        self.dropout = dropout
        
        self.W_q = L.Linear(d_model, in_size=d_model)
        self.W_k = L.Linear(d_model, in_size=d_model)
        self.W_v = L.Linear(d_model, in_size=d_model)
        self.W_o = L.Linear(d_model, in_size=d_model)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        
        # 线性变换
        q = self.W_q(query)  # (batch_size, seq_len, d_model)
        k = self.W_k(key)    # (batch_size, seq_len, d_model)
        v = self.W_v(value)  # (batch_size, seq_len, d_model)
        
        # 分割为多头
        q = q.reshape(batch_size, -1, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        k = k.reshape(batch_size, -1, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        v = v.reshape(batch_size, -1, self.n_head, self.d_k).transpose(0, 2, 1, 3)
        
        # 计算注意力分数
        scores = F.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.d_k)
        
        if mask is not None:
            # 应用掩码
            # 扩展mask以匹配scores的维度
            if mask.ndim < scores.ndim:
                mask = mask.reshape(*mask.shape, *(1,) * (scores.ndim - mask.ndim))
            
            # 使用一个大的负数填充被mask的位置
            # 确保mask是布尔类型
            mask_var = Variable(mask.astype(bool))
            scores = F.where(mask_var, scores, Variable(np.full(scores.shape, -1e9)))
        
        # 计算注意力权重
        attn_weights = F.softmax(scores, axis=-1)
        
        if self.dropout > 0:
            attn_weights = F.dropout(attn_weights, self.dropout)
        
        # 应用注意力权重
        context = F.matmul(attn_weights, v)
        
        # 重新组合多头
        # 先转置，然后reshape
        context = context.transpose(0, 2, 1, 3)
        context = context.reshape(batch_size, -1, self.d_model)
        
        # 输出线性变换
        output = self.W_o(context)
        
        return output


class PositionwiseFeedForward(Layer):
    """位置前馈网络"""
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.w_1 = L.Linear(d_ff, in_size=d_model)
        self.w_2 = L.Linear(d_model, in_size=d_ff)
        self.dropout = dropout
        
    def forward(self, x):
        x = self.w_1(x)
        x = F.relu(x)
        if self.dropout > 0:
            x = F.dropout(x, self.dropout)
        x = self.w_2(x)
        return x


class PositionalEncoding(Layer):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 创建位置编码表
        pe = np.zeros((max_len, d_model))
        position = np.arange(0, max_len).reshape(-1, 1)
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        pe = pe[np.newaxis, :]
        
        self.pe = Parameter(pe, name='pe')
        
    def forward(self, x):
        # x: (batch_size, seq_len, d_model)
        seq_len = x.shape[1]
        pe_slice = self.pe.data[:, :seq_len]
        x = x + Variable(pe_slice)
        return x


class EncoderLayer(Layer):
    """编码器层"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = L.LayerNorm(d_model)
        self.norm2 = L.LayerNorm(d_model)
        self.dropout = dropout
        
    def forward(self, x, mask=None):
        # 多头注意力
        attn_output = self.self_attn(x, x, x, mask)
        if self.dropout > 0:
            attn_output = F.dropout(attn_output, self.dropout)
        x = self.norm1(x + attn_output)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        if self.dropout > 0:
            ff_output = F.dropout(ff_output, self.dropout)
        x = self.norm2(x + ff_output)
        
        return x


class DecoderLayer(Layer):
    """解码器层"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.enc_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = L.LayerNorm(d_model)
        self.norm2 = L.LayerNorm(d_model)
        self.norm3 = L.LayerNorm(d_model)
        self.dropout = dropout
        
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 自注意力
        attn_output = self.self_attn(x, x, x, tgt_mask)
        if self.dropout > 0:
            attn_output = F.dropout(attn_output, self.dropout)
        x = self.norm1(x + attn_output)
        
        # 编码器-解码器注意力
        if enc_output is not None:
            attn_output = self.enc_attn(x, enc_output, enc_output, src_mask)
            if self.dropout > 0:
                attn_output = F.dropout(attn_output, self.dropout)
            x = self.norm2(x + attn_output)
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        if self.dropout > 0:
            ff_output = F.dropout(ff_output, self.dropout)
        x = self.norm3(x + ff_output)
        
        return x


class Encoder(Layer):
    """编码器"""
    def __init__(self, n_layers, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.layers = []
        for i in range(n_layers):
            layer = EncoderLayer(d_model, n_head, d_ff, dropout)
            setattr(self, 'layer_%d' % i, layer)
            self.layers.append(layer)
            
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x


class Decoder(Layer):
    """解码器"""
    def __init__(self, n_layers, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.layers = []
        for i in range(n_layers):
            layer = DecoderLayer(d_model, n_head, d_ff, dropout)
            setattr(self, 'layer_%d' % i, layer)
            self.layers.append(layer)
            
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return x


class Transformer(Layer):
    """Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, 
                 n_layers=6, d_model=512, n_head=8, d_ff=2048, 
                 dropout=0.1, max_len=5000):
        super().__init__()
        self.d_model = d_model
        
        # 词嵌入
        self.src_embed = L.EmbedID(src_vocab_size, d_model)
        self.tgt_embed = L.EmbedID(tgt_vocab_size, d_model)
        
        # 位置编码
        self.src_pos = PositionalEncoding(d_model, max_len)
        self.tgt_pos = PositionalEncoding(d_model, max_len)
        
        # 编码器和解码器
        self.encoder = Encoder(n_layers, d_model, n_head, d_ff, dropout)
        self.decoder = Decoder(n_layers, d_model, n_head, d_ff, dropout)
        
        # 输出投影
        self.output_projection = L.Linear(tgt_vocab_size, in_size=d_model)
        
        # 初始化参数
        self._init_params()
        
    def _init_params(self):
        """初始化参数"""
        pass
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # 编码器
        src_emb = self.src_embed(src) * np.sqrt(self.d_model)
        src_emb = self.src_pos(src_emb)
        enc_output = self.encoder(src_emb, src_mask)
        
        # 解码器
        tgt_emb = self.tgt_embed(tgt) * np.sqrt(self.d_model)
        tgt_emb = self.tgt_pos(tgt_emb)
        
        # 创建目标序列掩码（防止关注未来信息）
        if tgt_mask is None:
            tgt_seq_len = tgt.shape[1]
            tgt_mask = self._generate_square_subsequent_mask(tgt_seq_len)
        
        dec_output = self.decoder(tgt_emb, enc_output, src_mask, tgt_mask)
        
        # 输出投影
        output = self.output_projection(dec_output)
        
        return output
    
    def _generate_square_subsequent_mask(self, sz):
        """生成后续位置掩码，防止关注未来信息"""
        mask = np.triu(np.ones((sz, sz)), k=1).astype('uint8')
        return mask[np.newaxis, np.newaxis, :, :]
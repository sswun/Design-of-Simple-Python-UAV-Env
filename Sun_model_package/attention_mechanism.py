from .basic_model import *

# 该内容主要为注意力机制相关内容
def sequence_mask(X, valid_len, value=0):
    """在序列中屏蔽不相关的项"""
    if model_name == 'torch':
        maxlen = X.size(1)
        mask = dlmodel.arange((maxlen), dtype=dlmodel.float32,
                            device=X.device)[None, :] < valid_len[:, None]
        X[~mask] = value
        return X
    else:
        maxlen = X.shape[1]
        mask = dlmodel.arange((maxlen), dtype=dlmodel.float32)[None, :] < valid_len[:, None]
        Xtype = X.dtype
        X = X.astype(dlmodel.float32)
        X[~mask] = float(value)
        return X.astype(Xtype)

def masked_softmax(X, valid_lens):
    """通过在最后一个轴上掩蔽元素来执行softmax操作"""
    if model_name == 'torch':
    # X:3D张量，valid_lens:1D或2D张量
        if valid_lens is None:
            return nn.functional.softmax(X, dim=-1)
        else:
            shape = X.shape
            if valid_lens.dim() == 1:
                valid_lens = dlmodel.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape(-1)
            # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
            X = sequence_mask(X.reshape(-1, shape[-1]), valid_lens,
                                value=-1e6)
            return nn.functional.softmax(X.reshape(shape), dim=-1)
    else:
        if valid_lens is None:
            return nn.functional.softmax(X, axis=-1)
        else:
            shape = X.shape
            if valid_lens.dim() == 1:
                valid_lens = dlmodel.repeat_interleave(valid_lens, shape[1])
            else:
                valid_lens = valid_lens.reshape((-1,))
            # 最后一轴上被掩蔽的元素使用一个非常大的负值替换，从而其softmax输出为0
            X = sequence_mask(X.reshape((-1, shape[-1])), valid_lens,
                                value=-1e6)
            return nn.functional.softmax(X.reshape(shape), axis=-1)

class AdditiveAttention(nn.Module if model_name=='torch' else nn.Layer):
    """加性注意力"""
    def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
        super(AdditiveAttention, self).__init__(**kwargs)
        if model_name == 'torch':
            self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
            self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
            self.w_v = nn.Linear(num_hiddens, 1, bias=False)
            self.dropout = nn.Dropout(dropout)
        else:
            self.W_k = nn.Linear(key_size, num_hiddens, bias_attr=False)
            self.W_q = nn.Linear(query_size, num_hiddens, bias_attr=False)
            self.w_v = nn.Linear(num_hiddens, 1, bias_attr=False)
            self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values, valid_lens):
        if model_name == 'torch':
            queries, keys = self.W_q(queries), self.W_k(keys)
            # 在维度扩展后，
            # queries的形状：(batch_size，查询的个数，1，num_hidden)
            # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
            # 使用广播方式进行求和
            features = queries.unsqueeze(2) + keys.unsqueeze(1)
            features = dlmodel.tanh(features)
            # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
            # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
            scores = self.w_v(features).squeeze(-1)
            self.attention_weights = masked_softmax(scores, valid_lens)
            # values的形状：(batch_size，“键－值”对的个数，值的维度)
            return dlmodel.bmm(self.dropout(self.attention_weights), values)
        else:
            queries, keys = self.W_q(queries), self.W_k(keys)
            # 在维度扩展后，
            # queries的形状：(batch_size，查询的个数，1，num_hidden)
            # key的形状：(batch_size，1，“键－值”对的个数，num_hiddens)
            # 使用广播方式进行求和
            features = queries.unsqueeze(2) + keys.unsqueeze(1)
            features = dlmodel.tanh(features)
            # self.w_v仅有一个输出，因此从形状中移除最后那个维度。
            # scores的形状：(batch_size，查询的个数，“键-值”对的个数)
            scores = self.w_v(features).squeeze(-1)
            self.attention_weights = masked_softmax(scores, valid_lens)
            # values的形状：(batch_size，“键－值”对的个数，值的维度)
            return dlmodel.bmm(self.dropout(self.attention_weights), values)

class DotProductAttention(nn.Module if model_name=='torch' else nn.Layer):
    """缩放点积注意力"""
    def __init__(self, dropout, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)

    # queries的形状：(batch_size，查询的个数，d)
    # keys的形状：(batch_size，“键－值”对的个数，d)
    # values的形状：(batch_size，“键－值”对的个数，值的维度)
    # valid_lens的形状:(batch_size，)或者(batch_size，查询的个数)
    def forward(self, queries, keys, values, valid_lens=None):
        if model_name == 'torch':
            d = queries.shape[-1]
            # 设置transpose_b=True为了交换keys的最后两个维度
            scores = dlmodel.bmm(queries, keys.transpose(1,2)) / math.sqrt(d)
            self.attention_weights = masked_softmax(scores, valid_lens)
            return dlmodel.bmm(self.dropout(self.attention_weights), values)
        else:
            d = queries.shape[-1]
            # 设置transpose_b=True为了交换keys的最后两个维度
            scores = dlmodel.bmm(queries, keys.transpose((0,2,1))) / math.sqrt(d)
            self.attention_weights = masked_softmax(scores, valid_lens)
            return dlmodel.bmm(self.dropout(self.attention_weights), values)

class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    """带遮蔽的softmax交叉熵损失函数"""
    # pred的形状：(batch_size,num_steps,vocab_size)
    # label的形状：(batch_size,num_steps)
    # valid_len的形状：(batch_size,)
    def forward(self, pred, label, valid_len):
        weights = dlmodel.ones_like(label)
        weights = sequence_mask(weights, valid_len)
        self.reduction='none'
        if model_name == 'torch':
            unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
                pred.permute(0, 2, 1), label)
            weighted_loss = (unweighted_loss * weights).mean(dim=1)
        else:
            unweighted_loss = super(MaskedSoftmaxCELoss, self).forward(
                pred, label)
            weighted_loss = (unweighted_loss * weights).mean(axis=1)
        return weighted_loss

# 编码器
class Encoder(nn.Module if model_name=='torch' else nn.Layer):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs):
        super(Encoder, self).__init__(**kwargs)

    def forward(self, X, *args):
        raise NotImplementedError
    
# 解码器
class Decoder(nn.Module if model_name=='torch' else nn.Layer):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args):
        raise NotImplementedError

    def forward(self, X, state):
        raise NotImplementedError

# 序列到序列部分改进
class Seq2SeqEncoder(Encoder):
    """用于序列到序列学习的循环神经网络编码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__(**kwargs)
        # 嵌入层
        if model_name == 'torch':
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.rnn = nn.GRU(embed_size, num_hiddens, num_layers,
                            dropout=dropout)
        else:
            weight_ih_attr = dlmodel.ParamAttr(initializer=nn.initializer.XavierUniform())
            weight_hh_attr = dlmodel.ParamAttr(initializer=nn.initializer.XavierUniform())
            # 嵌入层
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.rnn = nn.GRU(embed_size, num_hiddens, num_layers, dropout=dropout,
                            time_major=True, weight_ih_attr=weight_ih_attr, weight_hh_attr=weight_hh_attr)

    def forward(self, X, *args):
        # 输出'X'的形状：(batch_size,num_steps,embed_size)
        X = self.embedding(X)
        # 在循环神经网络模型中，第一个轴对应于时间步
        if model_name == 'torch':
            X = X.permute(1, 0, 2)
        else:
            X = X.transpose([1, 0, 2])
        # 如果未提及状态，则默认为0
        output, state = self.rnn(X)
        # output的形状:(num_steps,batch_size,num_hiddens)
        # state的形状:(num_layers,batch_size,num_hiddens)
        return output, state

class Seq2SeqDecoder(Decoder):
    """用于序列到序列学习的循环神经网络解码器"""
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__(**kwargs)
        if model_name == 'torch':
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers,
                            dropout=dropout)
            self.dense = nn.Linear(num_hiddens, vocab_size)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
            weight_attr = dlmodel.ParamAttr(initializer=nn.initializer.XavierUniform())
            weight_ih_attr = dlmodel.ParamAttr(initializer=nn.initializer.XavierUniform())
            weight_hh_attr = dlmodel.ParamAttr(initializer=nn.initializer.XavierUniform())
            self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens, num_layers, dropout=dropout,
                            time_major=True, weight_ih_attr=weight_ih_attr,weight_hh_attr=weight_hh_attr)
            self.dense = nn.Linear(num_hiddens, vocab_size,weight_attr=weight_attr)

    def init_state(self, enc_outputs, *args):
        return enc_outputs[1]

    def forward(self, X, state):
        if model_name == 'torch':
            # 输出'X'的形状：(batch_size,num_steps,embed_size)
            X = self.embedding(X).permute(1, 0, 2)
            # 广播context，使其具有与X相同的num_steps
            context = state[-1].repeat(X.shape[0], 1, 1)
            X_and_context = dlmodel.cat((X, context), 2)
            output, state = self.rnn(X_and_context, state)
            output = self.dense(output).permute(1, 0, 2)
            # output的形状:(batch_size,num_steps,vocab_size)
            # state的形状:(num_layers,batch_size,num_hiddens)
            return output, state
        else:
            # 输出'X'的形状：(batch_size,num_steps,embed_size)
            X = self.embedding(X).transpose([1, 0, 2])
            # 广播context，使其具有与X相同的num_steps
            context = state[-1].tile([X.shape[0], 1, 1])
            X_and_context = dlmodel.concat((X, context), 2)
            output, state = self.rnn(X_and_context, state)
            output = self.dense(output).transpose([1, 0, 2])
            # output的形状:(batch_size,num_steps,vocab_size)
            # state[0]的形状:(num_layers,batch_size,num_hiddens)
            return output, state

# 合并编码器解码器
class EncoderDecoder(nn.Module if model_name=='torch' else nn.Layer):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)

# 注意力解码器
class AttentionDecoder(Decoder):
    """带有注意力机制解码器的基本接口"""
    def __init__(self, **kwargs):
        super(AttentionDecoder, self).__init__(**kwargs)

    @property
    def attention_weights(self):
        raise NotImplementedError

# 序列到序列注意力解码器
class Seq2SeqAttentionDecoder(AttentionDecoder):
    def __init__(self, vocab_size, embed_size, num_hiddens, num_layers,
                 dropout=0, **kwargs):
        super(Seq2SeqAttentionDecoder, self).__init__(**kwargs)
        self.attention = AdditiveAttention(
            num_hiddens, num_hiddens, num_hiddens, dropout)
        if model_name == 'torch':
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.rnn = nn.GRU(
                embed_size + num_hiddens, num_hiddens, num_layers,
                dropout=dropout)
            self.dense = nn.Linear(num_hiddens, vocab_size)
        else:
            self.embedding = nn.Embedding(vocab_size, embed_size)
            self.rnn = nn.GRU(embed_size + num_hiddens, num_hiddens,
                            num_layers, bias_ih_attr=True,
                            time_major=True, dropout=dropout)
            self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        # outputs的形状为(batch_size，num_steps，num_hiddens).
        # hidden_state的形状为(num_layers，batch_size，num_hiddens)
        outputs, hidden_state = enc_outputs
        if model_name == 'torch':
            return (outputs.permute(1, 0, 2), hidden_state, enc_valid_lens)
        else:
            return (outputs.transpose((1, 0, 2)), hidden_state, enc_valid_lens)

    def forward(self, X, state):
        # enc_outputs的形状为(batch_size,num_steps,num_hiddens).
        # hidden_state的形状为(num_layers,batch_size,
        # num_hiddens)
        if model_name == 'torch':
            enc_outputs, hidden_state, enc_valid_lens = state
            # 输出X的形状为(num_steps,batch_size,embed_size)
            X = self.embedding(X).permute(1, 0, 2)
            outputs, self._attention_weights = [], []
            for x in X:
                # query的形状为(batch_size,1,num_hiddens)
                query = dlmodel.unsqueeze(hidden_state[-1], dim=1)
                # context的形状为(batch_size,1,num_hiddens)
                context = self.attention(
                    query, enc_outputs, enc_outputs, enc_valid_lens)
                # 在特征维度上连结
                x = dlmodel.cat((context, dlmodel.unsqueeze(x, dim=1)), dim=-1)
                # 将x变形为(1,batch_size,embed_size+num_hiddens)
                out, hidden_state = self.rnn(x.permute(1, 0, 2), hidden_state)
                outputs.append(out)
                self._attention_weights.append(self.attention.attention_weights)
            # 全连接层变换后，outputs的形状为
            # (num_steps,batch_size,vocab_size)
            outputs = self.dense(dlmodel.cat(outputs, dim=0))
            return outputs.permute(1, 0, 2), [enc_outputs, hidden_state,
                                            enc_valid_lens]
        else:
            enc_outputs, hidden_state, enc_valid_lens = state
            # 输出X的形状为(num_steps,batch_size,embed_size)
            X = self.embedding(X).transpose((1, 0, 2))
            outputs, self._attention_weights = [], []
            for x in X:
                # query的形状为(batch_size,1,num_hiddens)
                query = dlmodel.unsqueeze(hidden_state[-1], axis=1)
                # context的形状为(batch_size,1,num_hiddens)
                context = self.attention(
                    query, enc_outputs, enc_outputs, enc_valid_lens)
                # 在特征维度上连结
                x = dlmodel.concat((context, dlmodel.unsqueeze(x, axis=1)), axis=-1)
                # 将x变形为(1,batch_size,embed_size+num_hiddens)
                out, hidden_state = self.rnn(x.transpose((1, 0, 2)), hidden_state)
                outputs.append(out)
                self._attention_weights.append(self.attention.attention_weights)
            # 全连接层变换后，outputs的形状为
            # (num_steps,batch_size,vocab_size)
            outputs = self.dense(dlmodel.concat(outputs, axis=0))
            return outputs.transpose((1, 0, 2)), [enc_outputs, hidden_state,
                                                enc_valid_lens]

    @property
    def attention_weights(self):
        return self._attention_weights

def transpose_qkv(X, num_heads):
    """为了多注意力头的并行计算而变换形状"""
    # 输入X的形状:(batch_size，查询或者“键－值”对的个数，num_hiddens)
    # 输出X的形状:(batch_size，查询或者“键－值”对的个数，num_heads，
    # num_hiddens/num_heads)
    if model_name == 'torch':
        X = X.reshape(X.shape[0], X.shape[1], num_heads, -1)

        # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        X = X.permute(0, 2, 1, 3)

        # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        return X.reshape(-1, X.shape[2], X.shape[3])
    else:
        X = X.reshape((X.shape[0], X.shape[1], num_heads, -1))

        # 输出X的形状:(batch_size，num_heads，查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        X = X.transpose((0, 2, 1, 3))

        # 最终输出的形状:(batch_size*num_heads,查询或者“键－值”对的个数,
        # num_hiddens/num_heads)
        return X.reshape((-1, X.shape[2], X.shape[3]))


def transpose_output(X, num_heads):
    """逆转transpose_qkv函数的操作"""
    if model_name == 'torch':
        X = X.reshape(-1, num_heads, X.shape[1], X.shape[2])
        X = X.permute(0, 2, 1, 3)
        return X.reshape(X.shape[0], X.shape[1], -1)
    else:
        X = X.reshape((-1, num_heads, X.shape[1], X.shape[2]))
        X = X.transpose((0, 2, 1, 3))
        return X.reshape((X.shape[0], X.shape[1], -1))

class MultiHeadAttention(nn.Module if model_name=='torch' else nn.Layer):
    """多头注意力"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 num_heads, dropout, bias=False, **kwargs):
        super(MultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.attention = DotProductAttention(dropout)
        if model_name == 'torch':
            self.W_q = nn.Linear(query_size, num_hiddens, bias=bias)
            self.W_k = nn.Linear(key_size, num_hiddens, bias=bias)
            self.W_v = nn.Linear(value_size, num_hiddens, bias=bias)
            self.W_o = nn.Linear(num_hiddens, num_hiddens, bias=bias)
        else:
            self.W_q = nn.Linear(query_size, num_hiddens, bias_attr=bias)
            self.W_k = nn.Linear(key_size, num_hiddens, bias_attr=bias)
            self.W_v = nn.Linear(value_size, num_hiddens, bias_attr=bias)
            self.W_o = nn.Linear(num_hiddens, num_hiddens, bias_attr=bias)

    def forward(self, queries, keys, values, valid_lens):
        # queries，keys，values的形状:
        # (batch_size，查询或者“键－值”对的个数，num_hiddens)
        # valid_lens　的形状:
        # (batch_size，)或(batch_size，查询的个数)
        # 经过变换后，输出的queries，keys，values　的形状:
        # (batch_size*num_heads，查询或者“键－值”对的个数，
        # num_hiddens/num_heads)
        queries = transpose_qkv(self.W_q(queries), self.num_heads)
        keys = transpose_qkv(self.W_k(keys), self.num_heads)
        values = transpose_qkv(self.W_v(values), self.num_heads)

        if valid_lens is not None:
            # 在轴0，将第一项（标量或者矢量）复制num_heads次，
            # 然后如此复制第二项，然后诸如此类。
            if model_name == 'torch':
                valid_lens = dlmodel.repeat_interleave(
                    valid_lens, repeats=self.num_heads, dim=0)
            else:
                valid_lens = dlmodel.repeat_interleave(
                    valid_lens, repeats=self.num_heads, axis=0)

        # output的形状:(batch_size*num_heads，查询的个数，
        # num_hiddens/num_heads)
        output = self.attention(queries, keys, values, valid_lens)

        # output_concat的形状:(batch_size，查询的个数，num_hiddens)
        output_concat = transpose_output(output, self.num_heads)
        return self.W_o(output_concat)

class PositionalEncoding(nn.Module if model_name=='torch' else nn.Layer):
    """位置编码"""
    def __init__(self, num_hiddens, dropout, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        if model_name == 'torch':
            self.P = dlmodel.zeros((1, max_len, num_hiddens))
            X = dlmodel.arange(max_len, dtype=dlmodel.float32).reshape(
                -1, 1) / dlmodel.pow(10000, dlmodel.arange(
                0, num_hiddens, 2, dtype=dlmodel.float32) / num_hiddens)
            self.P[:, :, 0::2] = dlmodel.sin(X)
            self.P[:, :, 1::2] = dlmodel.cos(X)
        else:
            self.P = dlmodel.zeros((1, max_len, num_hiddens))
            X = dlmodel.arange(max_len, dtype=dlmodel.float32).reshape(
                (-1, 1)) / dlmodel.pow(dlmodel.to_tensor([10000.0]), dlmodel.arange(
                0, num_hiddens, 2, dtype=dlmodel.float32) / num_hiddens)
            self.P[:, :, 0::2] = dlmodel.sin(X)
            self.P[:, :, 1::2] = dlmodel.cos(X)

    def forward(self, X):
        if model_name == 'torch':
            X = X + self.P[:, :X.shape[1], :].to(X.device)
            return self.dropout(X)
        else:
            X = X + self.P[:, :X.shape[1], :]
            return self.dropout(X)

class PositionWiseFFN(nn.Module if model_name=='torch' else nn.Layer):
    """基于位置的前馈网络"""
    def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))

class AddNorm(nn.Module if model_name=='torch' else nn.Layer):
    """残差连接后进行层规范化"""
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)

class EncoderBlock(nn.Module if model_name=='torch' else nn.Layer):
    """Transformer编码器块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, use_bias=False, **kwargs):
        super(EncoderBlock, self).__init__(**kwargs)
        self.attention = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout,
            use_bias)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(
            ffn_num_input, ffn_num_hiddens, num_hiddens)
        self.addnorm2 = AddNorm(norm_shape, dropout)

    def forward(self, X, valid_lens):
        Y = self.addnorm1(X, self.attention(X, X, X, valid_lens))
        return self.addnorm2(Y, self.ffn(Y))

#@save
class TransformerEncoder(Encoder):
    """Transformer编码器"""
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, use_bias=False, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        if model_name == 'torch':
            for i in range(num_layers):
                self.blks.add_module("block"+str(i),
                    EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                norm_shape, ffn_num_input, ffn_num_hiddens,
                                num_heads, dropout, use_bias))
        else:
            for i in range(num_layers):
                self.blks.add_sublayer(str(i),
                    EncoderBlock(key_size, query_size, value_size, num_hiddens,
                                norm_shape, ffn_num_input, ffn_num_hiddens,
                                num_heads, dropout, use_bias))

    def forward(self, X, valid_lens, *args):
        # 因为位置编码值在-1和1之间，
        # 因此嵌入值乘以嵌入维度的平方根进行缩放，
        # 然后再与位置编码相加。
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self.attention_weights = [None] * len(self.blks)
        for i, blk in enumerate(self.blks):
            X = blk(X, valid_lens)
            self.attention_weights[
                i] = blk.attention.attention.attention_weights
        return X

class DecoderBlock(nn.Module if model_name=='torch' else nn.Layer):
    """解码器中第i个块"""
    def __init__(self, key_size, query_size, value_size, num_hiddens,
                 norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
                 dropout, i, **kwargs):
        super(DecoderBlock, self).__init__(**kwargs)
        self.i = i
        self.attention1 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm1 = AddNorm(norm_shape, dropout)
        self.attention2 = MultiHeadAttention(
            key_size, query_size, value_size, num_hiddens, num_heads, dropout)
        self.addnorm2 = AddNorm(norm_shape, dropout)
        self.ffn = PositionWiseFFN(ffn_num_input, ffn_num_hiddens,
                                   num_hiddens)
        self.addnorm3 = AddNorm(norm_shape, dropout)

    def forward(self, X, state):
        enc_outputs, enc_valid_lens = state[0], state[1]
        # 训练阶段，输出序列的所有词元都在同一时间处理，
        # 因此state[2][self.i]初始化为None。
        # 预测阶段，输出序列是通过词元一个接着一个解码的，
        # 因此state[2][self.i]包含着直到当前时间步第i个块解码的输出表示
        if model_name == 'torch':
            if state[2][self.i] is None:
                key_values = X
            else:
                key_values = dlmodel.cat((state[2][self.i], X), axis=1)
            state[2][self.i] = key_values
            if self.training:
                batch_size, num_steps, _ = X.shape
                # dec_valid_lens的开头:(batch_size,num_steps),
                # 其中每一行是[1,2,...,num_steps]
                dec_valid_lens = dlmodel.arange(
                    1, num_steps + 1, device=X.device).repeat(batch_size, 1)
            else:
                dec_valid_lens = None

            # 自注意力
            X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
            Y = self.addnorm1(X, X2)
            # 编码器－解码器注意力。
            # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
            Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
            Z = self.addnorm2(Y, Y2)
            return self.addnorm3(Z, self.ffn(Z)), state
        else:
            if state[2][self.i] is None:
                key_values = X
            else:
                key_values = dlmodel.concat((state[2][self.i], X), axis=1)
            state[2][self.i] = key_values
            if self.training:
                batch_size, num_steps, _ = X.shape
                # dec_valid_lens的开头:(batch_size,num_steps),
                # 其中每一行是[1,2,...,num_steps]
                dec_valid_lens = dlmodel.arange(
                    1, num_steps + 1).tile((batch_size, 1))
            else:
                dec_valid_lens = None

            # 自注意力
            X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
            Y = self.addnorm1(X, X2)
            # 编码器－解码器注意力。
            # enc_outputs的开头:(batch_size,num_steps,num_hiddens)
            Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
            Z = self.addnorm2(Y, Y2)
            return self.addnorm3(Z, self.ffn(Z)), state

class TransformerDecoder(AttentionDecoder):
    def __init__(self, vocab_size, key_size, query_size, value_size,
                 num_hiddens, norm_shape, ffn_num_input, ffn_num_hiddens,
                 num_heads, num_layers, dropout, **kwargs):
        super(TransformerDecoder, self).__init__(**kwargs)
        self.num_hiddens = num_hiddens
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, num_hiddens)
        self.pos_encoding = PositionalEncoding(num_hiddens, dropout)
        self.blks = nn.Sequential()
        if model_name == 'torch':
            for i in range(num_layers):
                self.blks.add_module("block"+str(i),
                    DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                norm_shape, ffn_num_input, ffn_num_hiddens,
                                num_heads, dropout, i))
        else:
            for i in range(num_layers):
                self.blks.add_sublayer(str(i),
                    DecoderBlock(key_size, query_size, value_size, num_hiddens,
                                norm_shape, ffn_num_input, ffn_num_hiddens,
                                num_heads, dropout, i))
        self.dense = nn.Linear(num_hiddens, vocab_size)

    def init_state(self, enc_outputs, enc_valid_lens, *args):
        return [enc_outputs, enc_valid_lens, [None] * self.num_layers]

    def forward(self, X, state):
        X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
        self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
        for i, blk in enumerate(self.blks):
            X, state = blk(X, state)
            # 解码器自注意力权重
            self._attention_weights[0][
                i] = blk.attention1.attention.attention_weights
            # “编码器－解码器”自注意力权重
            self._attention_weights[1][
                i] = blk.attention2.attention.attention_weights
        return self.dense(X), state

    @property
    def attention_weights(self):
        return self._attention_weights

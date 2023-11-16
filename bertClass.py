'''
@Description: 
@Author: Panbo Hey
@Date: 2023-11-02 09:16:03
@LastEditTime: 2023-11-02 13:59:52
@LastEditors: Panbo Hey
'''

import torch
from torch import nn
from d2l import torch as d2l



class BERTEncoder(nn.Module):
    # BERT编码器
    def __init__(
            self,
            vocab_size,
            num_hiddens,
            norm_shape,
            ffn_num_input,
            ffn_num_hiddens,
            num_heads,
            num_layers,
            dropout,
            max_len = 1000,
            key_size = 768,
            query_size = 768,
            value_size = 768,
            **kwargs
    ):
        super(BERTEncoder, self).__init__(**kwargs)
        ##token 和 segement
        self.token_embedding = nn.Embedding(vocab_size, num_hiddens)
        self.segement_embedding = nn.Embedding(2, num_hiddens)
        self.blks = nn.Sequential()


        for i in range(num_layers):
            self.blks.add_module(f"{i}", d2l.EncoderBlock(
                key_size,
                query_size,
                value_size,
                num_hiddens,
                norm_shape,
                ffn_num_input,
                ffn_num_hiddens,
                num_heads,
                dropout,
                True
            ))


        #因为BERT中的位置编码是通过学习得到的，所以我们需要创建一个足够长的位置嵌入参数
        #注意到这里初始化使用的是随机数
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, num_hiddens))
    
    def forward(self, tokens, segments, valid_lens):
        #注意X的尺寸大小，始终为(批量大小， 最大序列长度， num_hiddens)
        X = self.token_embedding(tokens) + self.segement_embedding(segments)
        X = X +self.pos_embedding.data[: , :X.shape[1],: ]
        for blk in self.blks:
            X = blk(X, valid_lens)
            return X


class MaskLM(nn.Module):
    ##BERT的掩码模型
    def __init__(
            self,
            vocab_size,
            num_hiddens,
            num_inputs = 786,
            **kwargs  
                 ):
        super(MaskLM, self).__init__(**kwargs)

        #nn.Sequential相当于一个容器，构造了一个模型，当中是按照顺序，依次计算。
        #nn.Sequential() 可以允许将整个容器视为单个模块
        #forward()方法接收输入之后，nn.Sequential()按照内部模块的顺序自动依次计算并输出结果。
        self.mlp = nn.Sequential(
            nn.Linear(num_inputs, num_hiddens),
            nn.ReLU(),
            nn.LayerNorm(num_hiddens),
            nn.Linear(num_hiddens, vocab_size)
        )


    def forward(self, X, pred_positions):
        num_pred_positions = pred_positions.shape[1]
        pred_positions = pred_positions.reshape(-1)
        batch_size = X.shape[0]
        batch_idx = torch.arange(0 ,batch_size)
        ##batch_size = 2, num_pred_positions = 3
        ## 那么batch_idx 是np.array([0,0,0,1,1,1])
        batch_idx = torch.repeat_interleave(batch_idx, num_pred_positions)
        masked_X = X[batch_idx, pred_positions]
        masked_X = masked_X.reshape((batch_size, num_pred_positions, -1))
        mlm_Y_hat = self.nlp(masked_X)
        return mlm_Y_hat
        
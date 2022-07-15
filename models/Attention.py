import torch
import torch.nn as nn
import math


class Encoder_vox(torch.nn.Module):
    def __init__(self,):
        super(Encoder_vox,self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv3d(1, 32,kernel_size=5),
            torch.nn.MaxPool3d(2),
            torch.nn.ReLU()
        )
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv3d(32,64,kernel_size=3),
            torch.nn.MaxPool3d(2),
            torch.nn.ReLU()
        )
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv3d(64,128,kernel_size=3),
            torch.nn.ReLU()
        )
        self.layer4 = torch.nn.Sequential(
            torch.nn.Conv3d(128,256,kernel_size=3),
            torch.nn.ReLU()
        )

    def forward(self,x):
        bt_size = x.size(0)
        x = x.view(-1,1,32,32,32)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0),-1)
        x = x.view(bt_size,-1,x.size(-1))
        return x


class Attention(nn.Module):

    def __init__(self, hidden_size, num_attention_heads, dropout_prob=0.2):
        """
        假设 hidden_size = 128, num_attention_heads = 8, dropout_prob = 0.2
        即隐层维度为128，注意力头设置为8个
        """
        super(Attention, self).__init__()
        if hidden_size % num_attention_heads != 0:  # 整除
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (hidden_size, num_attention_heads))
        # 参数定义
        self.num_attention_heads = num_attention_heads  # 8
        self.attention_head_size = int(hidden_size / num_attention_heads)  # 16  每个注意力头的维度
        self.all_head_size = int(self.num_attention_heads * self.attention_head_size)
        self.encoder = Encoder_vox()

        # query, key, value 的线性变换（上述公式2）
        self.query = nn.Linear(12544, self.all_head_size)  # 128, 128
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)

        # dropout
        self.dropout = nn.Dropout(dropout_prob)

    def transpose_for_scores(self, x):
        # INPUT:  x'shape = [bs, seqlen, hid_size]  假设hid_size=128
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)  # [bs, seqlen, 8, 16]
        x = x.view(*new_x_shape)  #
        return x.permute(0, 2, 1, 3)  # [bs, 8, seqlen, 16]

    def forward(self, image_feature, prior_feature):
        image_feature = image_feature.view(image_feature.size(0), -1)
        prior_feature = self.encoder(prior_feature)
        image_feature = image_feature.unsqueeze(0)
        image_feature = image_feature.permute(1,0,2)

        # 线性变换
        mixed_query_layer = self.query(image_feature)  # [bs, seqlen, hid_size]
        mixed_key_layer = self.key(prior_feature)  # [bs, seqlen, hid_size]
        mixed_value_layer = self.value(prior_feature)  # [bs, seqlen, hid_size]

        query_layer = self.transpose_for_scores(mixed_query_layer)  # [bs, 8, seqlen, 16]
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)  # [bs, 8, seqlen, 16]

        # Take the dot product between "query" and "key" to get the raw attention scores.
        # 计算query与title之间的点积注意力分数，还不是权重（个人认为权重应该是和为1的概率分布）
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        # [bs, 8, seqlen, 16]*[bs, 8, 16, seqlen]  ==> [bs, 8, seqlen, seqlen]
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)  # [bs, 8, seqlen, seqlen]
        # 除以根号注意力头的数量，可看原论文公式，防止分数过大，过大会导致softmax之后非0即1
        # 将注意力转化为概率分布，即注意力权重
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # [bs, 8, seqlen, seqlen]
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # 矩阵相乘，[bs, 8, seqlen, seqlen]*[bs, 8, seqlen, 16] = [bs, 8, seqlen, 16]
        context_layer = torch.matmul(attention_probs, value_layer)  # [bs, 8, seqlen, 16]
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()  # [bs, seqlen, 8, 16]
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)  # [bs, seqlen, 128]
        context_layer = context_layer.view(*new_context_layer_shape)
        context_layer = torch.mean(context_layer, dim=1)
        return context_layer  # [bs, seqlen, 128] 得到输出


if __name__=='__main__':
    model = Attention(hidden_size=2048,num_attention_heads=2,dropout_prob=0.1)
    print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in model.parameters())))

    image_feature = torch.rand(16,1,256,7,7)
    prior_feature = torch.rand(16,6,32,32,32)
    out = model(image_feature,prior_feature)
    print(out.shape)
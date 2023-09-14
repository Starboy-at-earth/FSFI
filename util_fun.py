import torch, torchvision
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List
# from utils.HDFNet.MyModules import DenseTransLayer, DDPM
from utils.Trans_Block import Trans_Block


class PSPModule(nn.Module):
    # (1, 2, 3, 6)
    def __init__(self, sizes=(1, 3, 6, 8), dimension=2):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        elif dimension == 3:
            prior = nn.AdaptiveAvgPool3d(output_size=(size, size, size))
        return prior

    def forward(self, feats):
        n, c, _, _ = feats.size()
        priors = [stage(feats).view(n, c, -1) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center

def generate_coord(batch, height, width):
    # coord = Variable(torch.zeros(batch,8,height,width).cuda())
    #print(batch, height, width)
    xv, yv = torch.meshgrid([torch.arange(0,height), torch.arange(0,width)])
    #print(batch, height, width)
    xv_min = (xv.float()*2 - width)/width
    yv_min = (yv.float()*2 - height)/height
    xv_max = ((xv+1).float()*2 - width)/width
    yv_max = ((yv+1).float()*2 - height)/height
    xv_ctr = (xv_min+xv_max)/2
    yv_ctr = (yv_min+yv_max)/2
    hmap = torch.ones(height,width)*(1./height)
    wmap = torch.ones(height,width)*(1./width)
    coord = torch.autograd.Variable(torch.cat([xv_min.unsqueeze(0), yv_min.unsqueeze(0),\
        xv_max.unsqueeze(0), yv_max.unsqueeze(0),\
        xv_ctr.unsqueeze(0), yv_ctr.unsqueeze(0),\
        hmap.unsqueeze(0), wmap.unsqueeze(0)], dim=0).cuda())
    coord = coord.unsqueeze(0).repeat(batch,1,1,1)
    return coord

class WordVisualAttention(nn.Module):
  def __init__(self, input_dim):
    super(WordVisualAttention, self).__init__()
    # initialize pivot
    self.visual = nn.Conv2d(input_dim, input_dim, kernel_size=1)

  def forward(self, context, visual, input_labels):
    """
    Inputs:
    - context : Variable float (batch, seq_len, input_dim)
    - embedded: Variable float (batch, seq_len, word_vec_size)
    - input_labels: Variable long (batch, seq_len)
    Outputs:
    - attn    : Variable float (batch, seq_len)
    - weighted_emb: Variable float (batch, word_vec_size)
    """
    visual = self.visual(visual)
    b_size, n_channel, h, w = visual.shape
    visual = visual.view(b_size, n_channel, h*w)
    attn = torch.bmm(context, visual)
    attn = F.softmax(attn, dim=1)  # (batch, seq_len), attn.sum(1) = 1.

    # mask zeros
    is_not_zero = (input_labels!=0).float()
    is_not_zero = is_not_zero.view(is_not_zero.size(0), is_not_zero.size(1), 1).repeat(1, 1, h*w)
    attn = attn * is_not_zero
    attn = attn / attn.sum(1).view(attn.size(0), 1, attn.size(2)).repeat(1, attn.size(1), 1)

    # compute weighted lang
    weighted_emb = torch.bmm(context.permute(0, 2, 1), attn)
    weighted_emb = weighted_emb.view(weighted_emb.size(0), weighted_emb.size(1), h, w)

    return weighted_emb

class VisualWordAttention(nn.Module):
  def __init__(self, input_dim):
    super(VisualWordAttention, self).__init__()
    # initialize pivot
    self.visual = nn.Conv2d(input_dim, input_dim, kernel_size=1)
    self.norm = nn.LayerNorm(input_dim)

  def forward(self, context, visual):
    """
    Inputs:
    - context : Variable float (batch, seq_len, input_dim)
    - embedded: Variable float (batch, seq_len, word_vec_size)
    - input_labels: Variable long (batch, seq_len)
    Outputs:
    - attn    : Variable float (batch, seq_len)
    - weighted_emb: Variable float (batch, word_vec_size)
    """
    visual = self.visual(visual)
    b_size, n_channel, h, w = visual.shape
    visual = visual.view(b_size, n_channel, h*w)
    attn = torch.bmm(context, visual)
    attn = F.softmax(attn, dim=2)  # (batch, seq_len), attn.sum(1) = 1.

    # compute weighted visual
    weighted_emb = torch.bmm(attn, visual.permute(0, 2, 1))
    #weighted_emb = weighted_emb.view(weighted_emb.size(0), weighted_emb.size(1), h, w)
    weighted_emb = self.norm(weighted_emb)

    return weighted_emb

class RNNEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_size, word_vec_size, hidden_size, bidirectional=False,
               input_dropout_p=0, dropout_p=0, n_layers=1, rnn_type='lstm', variable_lengths=True, bert_embededing=None):
        super(RNNEncoder, self).__init__()
        self.variable_lengths = variable_lengths
        self.bert_embededing = bert_embededing
        if bert_embededing is not None:
            self.embedding = bert_embededing
        else:
            self.embedding = nn.Embedding(vocab_size, word_embedding_size)
        self.input_dropout = nn.Dropout(input_dropout_p)
        self.mlp = nn.Sequential(nn.Linear(word_embedding_size, word_vec_size),
                                 nn.ReLU())
        self.rnn_type = rnn_type
        self.rnn = getattr(nn, rnn_type.upper())(word_vec_size, hidden_size, n_layers,
                                                 batch_first=True,
                                                 bidirectional=bidirectional,
                                                 dropout=dropout_p)
        self.num_dirs = 2 if bidirectional else 1

    def forward(self, input_labels):
        """
        Inputs:
        - input_labels: Variable long (batch, seq_len)
        Outputs:
        - output  : Variable float (batch, max_len, hidden_size * num_dirs)
        - hidden  : Variable float (batch, num_layers * num_dirs * hidden_size)
        - embedded: Variable float (batch, max_len, word_vec_size)
        """
        if self.variable_lengths:
            input_lengths = (input_labels!=0).sum(1)  # Variable (batch, )

            # make ixs
            input_lengths_list = input_lengths.data.cpu().numpy().tolist()
            sorted_input_lengths_list = np.sort(input_lengths_list)[::-1].tolist() # list of sorted input_lengths
            sort_ixs = np.argsort(input_lengths_list)[::-1].tolist() # list of int sort_ixs, descending
            s2r = {s: r for r, s in enumerate(sort_ixs)} # O(n)
            recover_ixs = [s2r[s] for s in range(len(input_lengths_list))]  # list of int recover ixs
            assert max(input_lengths_list) == input_labels.size(1)

            # move to long tensor
            sort_ixs = input_labels.data.new(sort_ixs).long()  # Variable long
            recover_ixs = input_labels.data.new(recover_ixs).long()  # Variable long

            # sort input_labels by descending order
            input_labels = input_labels[sort_ixs]

        # embed
        if self.bert_embededing is not None:
            embedded = self.embedding(input_labels) # (n, seq_len, word_embedding_size)
        else:
            embedded = self.embedding(input_labels)  # (n, seq_len, word_embedding_size)
        embedded = self.input_dropout(embedded)  # (n, seq_len, word_embedding_size)
        embedded = self.mlp(embedded)            # (n, seq_len, word_vec_size)
        if self.variable_lengths:
            embedded = nn.utils.rnn.pack_padded_sequence(embedded, torch.as_tensor(sorted_input_lengths_list, dtype=torch.int64).cpu(), batch_first=True)

        # forward rnn
        self.rnn.flatten_parameters() #不加这一行会报warning
        output, hidden = self.rnn(embedded)

        # recover
        if self.variable_lengths:

            # embedded (batch, seq_len, word_vec_size)
            embedded, _ = nn.utils.rnn.pad_packed_sequence(embedded, batch_first=True)
            embedded = embedded[recover_ixs]

            # recover rnn
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)  # (batch, max_len, hidden)
            output = output[recover_ixs]

            # # recover hidden
            # if self.rnn_type == 'lstm':
            #     hidden = hidden[0]
            # hidden = hidden[:, recover_ixs, :]  # (num_layers * num_dirs, batch, hidden_size)
            # hidden = hidden.transpose(0, 1).contiguous()  # (batch, num_layers * num_dirs, hidden_size)
            # hidden = hidden.view(hidden.size(0), -1)  # (batch, num_layers * num_dirs * hidden_size)
        sent_output = []
        for ii in range(output.shape[0]):
            sent_output.append(output[ii, int(input_lengths_list[ii] - 1), :])

        return output, torch.stack(sent_output, dim=0), embedded

class Cross_AttentionBlock(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, scale=1):
        super(Cross_AttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.query_channels = key_channels
        self.value_channels = value_channels
        self.wordvisual = WordVisualAttention(in_channels)
        self.f_key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1)
        self.f_query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.query_channels, kernel_size=1)
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1)
        self.W1 = nn.Conv2d(in_channels=self.in_channels*2, out_channels=self.out_channels, kernel_size=1)
        self.W2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)


    def forward(self, key, query, word_id):
        # key: visual
        # query: language
        query = self.wordvisual(query, key, word_id)
        ############################################################
        tmp_key, tmp_query = key, query
        batch_size, h, w = key.size(0), key.size(2), key.size(3)
        key = self.f_key(tmp_key).view(batch_size, self.key_channels, -1)
        query = self.f_query(tmp_query).view(batch_size, self.query_channels, -1)
        query = query.permute(0, 2, 1)
        value = self.f_value(tmp_key).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        W1 = self.W1(torch.cat([context, tmp_query], dim=1))
        W2 = self.W2(context) + W1
        return W2


class DepthDC3x3_3(nn.Module):
    def __init__(self, in_xC, in_yC, out_C, down_factor=4):
        """DepthDC3x3_3，利用nn.Unfold实现的动态卷积模块

        Args:
            in_xC (int): 第一个输入的通道数
            in_yC (int): 第二个输入的通道数
            out_C (int): 最终输出的通道数
            down_factor (int): 用来降低卷积核生成过程中的参数量的一个降低通道数的参数
        """
        super(DepthDC3x3_3, self).__init__()
        self.fuse = nn.Conv2d(in_xC, out_C, 3, 1, 1)
        self.kernel_size = 3
        self.gernerate_kernel = nn.Sequential(
            nn.Conv2d(in_yC, in_yC, 3, 1, 1),
            #DenseFuseV3(in_yC, in_yC, k=down_factor),
            nn.Conv2d(in_yC, in_xC * self.kernel_size ** 2, 1),
        )
        self.unfold = nn.Unfold(kernel_size=3, dilation=1, padding=1, stride=1)

    def forward(self, x, y):
        N, xC, xH, xW = x.size()
        kernel = self.gernerate_kernel(y).reshape(
            [N, xC, self.kernel_size ** 2, xH, xW]
        )
        unfold_x = self.unfold(x).reshape([N, xC, -1, xH, xW])
        result = (unfold_x * kernel).sum(2)
        return self.fuse(result)



class CoattentionModel(nn.Module):
    def __init__(self, in_channel=512, out_channel=512, all_dim=40 * 40):  # 473./8=60
        super(CoattentionModel, self).__init__()
        self.linear_e = nn.Conv2d(in_channel, 256, kernel_size=1, padding=0, bias=False)
        self.linear_q = nn.Conv2d(in_channel, 256, kernel_size=1, padding=0, bias=False)
        #self.channel = all_channel
        self.dim = all_dim
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, bias=False)
        # self.prelu = nn.ReLU(inplace=True)
        # self.main_classifier1 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias=True)
        # self.main_classifier2 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias=True)
        # self.softmax = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                # init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # init.xavier_normal(m.weight.data)
                # m.bias.data.fill_(0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()

    def forward(self, exemplar, query):  # 注意input2 可以是多帧图像

        fea_size = query.size()[2:]
        all_dim = fea_size[0] * fea_size[1]
        query_corr = self.linear_q(query)
        exemplar_corr = self.linear_e(exemplar)  #
        exemplar_flat = exemplar_corr.view(-1, exemplar_corr.size()[1], all_dim)  # N,C,H*W
        query_flat = query_corr.view(-1, query_corr.size()[1], all_dim)
        exemplar_t = torch.transpose(exemplar_flat, 1, 2).contiguous()  # batch size x dim x num
        A = torch.bmm(exemplar_t, query_flat)
        A1 = F.softmax(A.clone(), dim=1).transpose(1,2) #
        B = F.softmax(torch.transpose(A, 1, 2), dim=1).transpose(1,2)
        exemplar_flat = exemplar.view(-1, exemplar.size()[1], all_dim)  # N,C,H*W
        query_flat = query.view(-1, query.size()[1], all_dim)
        query_att = torch.bmm(exemplar_flat, A1).contiguous()  # 注意我们这个地方要不要用交互以及Residual的结构
        exemplar_att = torch.bmm(query_flat, B).contiguous()

        input1_att = exemplar_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input2_att = query_att.view(-1, query.size()[1], fea_size[0], fea_size[1])
        input1_att = self.conv1(input1_att)
        input2_att = self.conv2(input2_att)
        return torch.cat([input1_att, input2_att],dim=1)  # shape: NxCx


class BCAM(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, dim=40*40):
        super(BCAM, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.query_channels = key_channels
        self.value_channels = value_channels
        self.wordvisual = WordVisualAttention(in_channels)
        self.f_key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1)
        self.f_query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.query_channels, kernel_size=1)
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1)
        self.dim_w = nn.Conv2d(in_channels=self.key_channels, out_channels=self.dim, kernel_size=1)
        self.W1 = nn.Conv2d(in_channels=self.in_channels*2, out_channels=self.out_channels, kernel_size=1)
        self.W2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)


    def forward(self, key, query, word_id):
        # key: visual
        # query: language
        query = self.wordvisual(query, key, word_id)
        ############################################################
        tmp_key, tmp_query = key, query
        batch_size, h, w = key.size(0), key.size(2), key.size(3)
        key = self.f_key(tmp_key)#.view(batch_size, self.key_channels, -1)
        query = self.f_query(tmp_query)#.view(batch_size, self.query_channels, -1)
        sim_map = F.softmax(self.dim_w(torch.tanh(key + query)), dim=1).view(batch_size, self.dim, -1)
        sim_map = sim_map.permute(0, 2, 1)
        value = self.f_value(tmp_key).view(batch_size, self.value_channels, -1)
        value = value.permute(0, 2, 1)

        # sim_map = torch.matmul(query, key)
        # sim_map = (self.key_channels**-.5) * sim_map
        # sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        W1 = self.W1(torch.cat([context, tmp_query], dim=1))
        W2 = self.W2(context) + W1
        return W2

class Asy_BCAM(nn.Module):
    '''
    The basic implementation for self-attention block/non-local block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        value_channels    : the dimension after the value transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
        position-aware context features.(w/o concate or add with the input)
    '''
    def __init__(self, in_channels, key_channels, value_channels, out_channels=None, dim=40*40, psp_size=(1,3,6,8)):
        super(Asy_BCAM, self).__init__()
        self.dim = dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.key_channels = key_channels
        self.query_channels = key_channels
        self.value_channels = value_channels
        self.wordvisual = WordVisualAttention(in_channels)
        self.f_key = nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels, kernel_size=1)
        self.f_query = nn.Conv2d(in_channels=self.in_channels, out_channels=self.query_channels, kernel_size=1)
        self.f_value = nn.Conv2d(in_channels=self.in_channels, out_channels=self.value_channels, kernel_size=1)
        self.dim_w = nn.Conv2d(in_channels=self.key_channels, out_channels=self.dim, kernel_size=1)
        self.W1 = nn.Conv2d(in_channels=self.in_channels*2, out_channels=self.out_channels, kernel_size=1)
        self.W2 = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)
        self.psp = PSPModule(psp_size)


    def forward(self, key, query, word_id):
        # key: visual
        # query: language
        query = self.wordvisual(query, key, word_id)
        ############################################################
        tmp_key, tmp_query = key, query
        batch_size, h, w = key.size(0), key.size(2), key.size(3)
        key = self.psp(self.f_key(tmp_key))
        query = self.f_query(tmp_query).view(batch_size, self.key_channels, -1).permute(0, 2, 1)
        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)
        value = self.psp(self.f_value(tmp_key)).permute(0, 2, 1)

        # sim_map = torch.matmul(query, key)
        # sim_map = (self.key_channels**-.5) * sim_map
        # sim_map = F.softmax(sim_map, dim=-1)

        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.value_channels, h, w)
        W1 = self.W1(torch.cat([context, tmp_query], dim=1))
        W2 = self.W2(context) + W1
        return W2



# class Cross_Frame(nn.Module):
#     '''
#     The basic implementation for self-attention block/non-local block
#     Input:
#         N X C X H X W
#     Parameters:
#         in_channels       : the dimension of the input feature map
#         key_channels      : the dimension after the key/query transform
#         value_channels    : the dimension after the value transform
#         scale             : choose the scale to downsample the input feature maps (save memory cost)
#     Return:
#         N X C X H X W
#         position-aware context features.(w/o concate or add with the input)
#     '''
#     def __init__(self, cur_in_channels, pre_in_channels, mid_channels, out_channels=None):
#         super(Cross_Frame, self).__init__()
#         self.cur_in_channels = cur_in_channels
#         self.pre_in_channels = pre_in_channels
#         self.out_channels = out_channels
#         self.mid_channels = mid_channels
#         self.current_f = nn.Conv2d(self.cur_in_channels, mid_channels, kernel_size=1, bias=False)
#         self.previous_f = nn.Linear(self.pre_in_channels, mid_channels, bias=False)
#         self.fuse = nn.Conv2d(in_channels=512, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
#         #self.DenseTrans = DenseTransLayer(512, 64)
#         self.selfdc = DDPM(512, 512, self.out_channels, 3, 4)
#         #self.trans = nn.Conv2d(512, 64, kernel_size=1, bias=False)
#
#
#     def forward(self, visual, context, input_labels):
#         # key: visual
#         # query: language
#         ############################################################
#         """
#             Inputs:
#             - context : Variable float (batch, seq_len, input_dim)
#             - embedded: Variable float (batch, seq_len, word_vec_size)
#             - input_labels: Variable long (batch, seq_len)
#             Outputs:
#             - attn    : Variable float (batch, seq_len)
#             - weighted_emb: Variable float (batch, word_vec_size)
#             """
#         visual = self.current_f(visual)
#         context = self.previous_f(context)
#         b_size, n_channel, h, w = visual.shape
#         ######
#         context = context.reshape(b_size, -1, context.size(2))
#         input_labels = input_labels.reshape(b_size, -1)
#         ######
#         #visual = visual.reshape(b_size, n_channel, h * w)
#         attn = torch.bmm(context, visual.reshape(b_size, n_channel, h * w))
#         attn = F.softmax(attn, dim=1)  # (batch, seq_len), attn.sum(1) = 1.
#
#         # mask zeros
#         is_not_zero = (input_labels != 0).float()
#         is_not_zero = is_not_zero.view(is_not_zero.size(0), is_not_zero.size(1), 1).repeat(1, 1, h * w)
#         attn = attn * is_not_zero
#         attn = attn / attn.sum(1).view(attn.size(0), 1, attn.size(2)).repeat(1, attn.size(1), 1)
#
#         # compute weighted lang
#         weighted_emb = torch.bmm(context.permute(0, 2, 1), attn)
#         weighted_emb = weighted_emb.view(weighted_emb.size(0), weighted_emb.size(1), h, w)
#
#         #visual_aux = self.DenseTrans(visual, weighted_emb)
#         visual = self.selfdc(visual, weighted_emb)
#
#
#         #outputs = self.fuse(torch.cat([visual, weighted_emb], dim=1))
#
#         return visual

# class Cross_Frame(nn.Module):
#     '''
#     The basic implementation for self-attention block/non-local block
#     Input:
#         N X C X H X W
#     Parameters:
#         in_channels       : the dimension of the input feature map
#         key_channels      : the dimension after the key/query transform
#         value_channels    : the dimension after the value transform
#         scale             : choose the scale to downsample the input feature maps (save memory cost)
#     Return:
#         N X C X H X W
#         position-aware context features.(w/o concate or add with the input)
#     '''
#     def __init__(self, in_channels, mid_channels, out_channels=None):
#         super(Cross_Frame, self).__init__()
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.mid_channels = mid_channels
#         self.reduce_c = nn.Conv2d(self.in_channels, mid_channels, kernel_size=1, bias=False)
#         #self.previous_f = nn.Linear(self.pre_in_channels, mid_channels, bias=False)
#         #self.fuse = nn.Conv2d(in_channels=mid_channels*2, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)
#         #self.DenseTrans = DenseTransLayer(512, 64)
#         self.selfdc = DDPM(512, 512, self.out_channels, 3, 4)
#         #self.trans = nn.Conv2d(512, 64, kernel_size=1, bias=False)
#         self.visualword_ref = VisualWordAttention(512)
#         self.visualword_cur = VisualWordAttention(512)
#         self.lang_encoder_ref = Trans_Block(dim=mid_channels, num_heads=8, mlp_drop=0., attn_drop=0.1)
#         self.lang_encoder_cur = Trans_Block(dim=mid_channels, num_heads=8, mlp_drop=0., attn_drop=0.1)
#         self.wordvisual = WordVisualAttention(512)


#     def forward(self, inputs, tmp_embedded, padding_mask, B, search_num, input_labels):
#         # key: visual
#         # query: language
#         ############################################################
#         """
#             Inputs:
#             - context : Variable float (batch, seq_len, input_dim)
#             - embedded: Variable float (batch, seq_len, word_vec_size)
#             - input_labels: Variable long (batch, seq_len)
#             Outputs:
#             - attn    : Variable float (batch, seq_len)
#             - weighted_emb: Variable float (batch, word_vec_size)
#             """
#         _, C, H, W = inputs.size()
#         inputs = self.reduce_c(inputs)
#         ######
#         inputs = F.normalize(inputs, p=2, dim=1)
#         ######
#         cur = inputs[0:B]
#         ref = inputs[B:(B + B * search_num)]
#         ref = ref.reshape(B, search_num, self.mid_channels, H, W)
#         ######
#         # cur_emb = tmp_embedded[0:B]
#         # ref_emb = tmp_embedded[B:(B + B * search_num)]
#         # ref_emb = ref_emb.reshape(B, search_num, ref_emb.size(1), ref_emb.size(2))
#         ######
#         embedded = F.normalize(tmp_embedded[0:B], p=2, dim=2)
#         padding_mask = padding_mask[0:B]
#         for iii in range(search_num):
#             embedded = self.visualword_ref(embedded, ref[:, iii, :]) + embedded
#             embedded = self.lang_encoder_ref(embedded, padding_mask=padding_mask)
#         ######
#         embedded = self.visualword_cur(embedded, cur) + embedded
#         embedded = self.lang_encoder_cur(embedded, padding_mask=padding_mask)
#         ######
#         weighted_emb = self.wordvisual(embedded, cur, input_labels[0:B])

#         visual = self.selfdc(cur, weighted_emb)


#         #outputs = self.fuse(torch.cat([cur, weighted_emb], dim=1))

#         return visual

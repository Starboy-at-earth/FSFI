import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import numpy as np
from data.config import cfg
from utils import timer
import math
from pytorch_pretrained_bert.modeling import BertModel
import time

from timm.models import create_model
from resnest.torch import resnest50_fast_2s1x64d
from buffer.mutual_attention import VisualWordAttention
from buffer.dynamic_conv import MultiHeadsDynamicConvolution
from decoder import MAED

from resnest.torch import resnest101

# These are in BGR and are for ImageNet
MEANS = (103.94, 116.78, 123.68)
STD   = (57.38, 57.12, 58.40)
means = np.array(MEANS)[np.newaxis,:,np.newaxis,np.newaxis]
std = np.array(STD)[np.newaxis,:,np.newaxis,np.newaxis]

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


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

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return self.drop_path(x, self.drop_prob, self.training)

    def drop_path(self, x, drop_prob: float = 0., training: bool = False):
        """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
        This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
        the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
        See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
        changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
        'survival rate' as the argument.
        """
        if drop_prob == 0. or not training:
            return x
        keep_prob = 1 - drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Trans_Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, mlp_drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop)
        # self.attn = Attention(
        #     dim,
        #     num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #     attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=mlp_drop)

    def forward(self, x, padding_mask):
        B, L, C = x.size()
        x = self.norm1(x)
        norm1_x = x.permute(1, 0, 2).contiguous()
        x = x + self.drop_path(self.attn(query=norm1_x, key=norm1_x, value=norm1_x, key_padding_mask=padding_mask)[0].reshape(B, L, C))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        return x

def Upsample(x, size):
    """
    Wrapper Around the Upsample Call
    """
    return nn.functional.interpolate(x, size=size, mode='bilinear',
                                     align_corners=False)
class ConvBNRelu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, G=32, use_relu=True):
        super(ConvBNRelu, self).__init__()
        self.use_relu = use_relu
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        if self.use_relu:
            self.relu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.img_bn = nn.BatchNorm2d(3)
        #---------------------------------------------------------------------------------------
        # resnet = resnest50_fast_2s1x64d(pretrained=True, dilation=2)
        resnet = resnest101(pretrained=True, dilation=2)
        self.layer0 = nn.Sequential(resnet.conv1,
                                    resnet.bn1,
                                    resnet.relu,
                                    resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.embed_dims = [128, 256, 512, 1024, 2048]       
        #----------------------------------------------------------------------------------------
        bt = BertModel.from_pretrained('bert-base-uncased')
        self.BertEmbededing = list(bt.children())[0]
        self.layers = nn.ModuleList(bt.encoder.layer.children())

        #----------------------------------------------------------------------------------------
        cfg.dcm_c = 512
        self.down_vision_layer1 = ConvBNRelu(self.embed_dims[1], cfg.dcm_c, kernel_size=1, stride=1, padding=0)
        self.up_vision_layer1 = ConvBNRelu(cfg.dcm_c, self.embed_dims[1], kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.embed_dims[1])

        self.down_vision_layer2 = ConvBNRelu(self.embed_dims[2], cfg.dcm_c, kernel_size=1, stride=1, padding=0)
        self.up_vision_layer2 = ConvBNRelu(cfg.dcm_c, self.embed_dims[2], kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(self.embed_dims[2])

        self.down_vision_layer3 = ConvBNRelu(self.embed_dims[3], cfg.dcm_c, kernel_size=1, stride=1, padding=0)
        self.up_vision_layer3 = ConvBNRelu(cfg.dcm_c, self.embed_dims[3], kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(self.embed_dims[3])

        self.down_vision_layer4 = ConvBNRelu(self.embed_dims[4], cfg.dcm_c, kernel_size=1, stride=1, padding=0)
        self.up_vision_layer4 = ConvBNRelu(cfg.dcm_c, self.embed_dims[4], kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(self.embed_dims[4])

        #------------------------------------------------------------------------------------------
        self.down_lang_layer1 = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU())

        self.down_lang_layer2 = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU())
       
        self.down_lang_layer3 = nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU())

        self.down_lang_layer4 = torch.nn.Sequential(
            nn.Linear(768, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.ReLU())
        L = 100 
        #--------------------------------------Global Enhancement-----------------------------------
        self.dynamic_convolution1=MultiHeadsDynamicConvolution(512, L)
        self.dynamic_convolution2=MultiHeadsDynamicConvolution(512, L)
        self.dynamic_convolution3=MultiHeadsDynamicConvolution(512, L)
        self.dynamic_convolution4=MultiHeadsDynamicConvolution(512, L)


        #---------------------------------------------FPN-------------------------------------------
        self.reduced_c4 = ConvBNRelu(self.embed_dims[4], self.embed_dims[3], kernel_size=1, stride=1, padding=0)
        self.reduced_c3 = ConvBNRelu(self.embed_dims[3], self.embed_dims[2], kernel_size=1, stride=1, padding=0)
        self.reduced_c2 = ConvBNRelu(self.embed_dims[2], self.embed_dims[1], kernel_size=1, stride=1, padding=0)
        self.reduced_c1 = ConvBNRelu(self.embed_dims[1], self.embed_dims[0], kernel_size=1, stride=1, padding=0)
        self.out_c4 = ConvBNRelu(self.embed_dims[4], 1, kernel_size=1, stride=1, padding=0)
        self.out_c3 = ConvBNRelu(self.embed_dims[3], 1, kernel_size=1, stride=1, padding=0)
        self.out_c2 = ConvBNRelu(self.embed_dims[2], 1, kernel_size=1, stride=1, padding=0)
        self.out_c1 = ConvBNRelu(self.embed_dims[1], 1, kernel_size=1, stride=1, padding=0)
        self.reduced_c0 = ConvBNRelu(self.embed_dims[0], 1, kernel_size=1, stride=1, padding=0)
        self.decoder = MAED()

        self.compression_one = nn.Conv2d(in_channels = 256, out_channels=512, kernel_size=8, stride=8) 
        self.compression_two = nn.Conv2d(in_channels = 512, out_channels=512, kernel_size=4, stride=4)
        self.compression_three = nn.Conv2d(in_channels = 1024, out_channels=512, kernel_size=2, stride=2) 
        self.compression_four = nn.Conv2d(in_channels = 2048, out_channels=512, kernel_size=2, stride=2) 
 

    def save_weights(self, path, epoch, iteration, cur_lr, optimizer):
        """ Saves the model's weights using compression because the file sizes were getting too big. """

        torch.save({'epoch': epoch, 
                    'iteration': iteration,
                    'state_dict': self.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'cur_lr': cur_lr}, 
                    path)
        
        print('saved ', path)

    def load_weights(self, path):
        """ Loads weights from a compressed save file. """
        state_dict = torch.load(path)

        # For backward compatability, remove these (the new variable is called layers)
        for key in list(state_dict.keys()):
            if key.startswith('backbone.layer') and not key.startswith('backbone.layers'):
                del state_dict[key]

            # Also for backward compatibility with v1.0 weights, do this check
            if key.startswith('fpn.downsample_layers.'):
                if cfg.fpn is not None and int(key.split('.')[2]) >= cfg.fpn.num_downsample:
                    del state_dict[key]
        self.load_state_dict(state_dict['state_dict'], strict=False)


    def init_weights(self, backbone_path):
        """ Initialize weights for training. """
        # Initialize the backbone with the pretrained weights.
        #self.backbone.init_backbone(backbone_path)
        self.load_state_dict(torch.load('yolact_darknet53_54_800000.pth'), strict=False)

        conv_constants = getattr(nn.Conv2d(1, 1, 1), '__constants__')

        # Quick lambda to test if one list contains the other
        def all_in(x, y):
            for _x in x:
                if _x not in y:
                    return False
            return True

        # Initialize the rest of the conv layers with xavier
        for name, module in self.named_modules():
            # See issue #127 for why we need such a complicated condition if the module is a WeakScriptModuleProxy
            # Broke in 1.3 (see issue #175), WeakScriptModuleProxy was turned into just ScriptModule.
            # Broke in 1.4 (see issue #292), where RecursiveScriptModule is the new star of the show.
            # Note that this might break with future pytorch updates, so let me know if it does
            is_script_conv = False
            if 'Script' in type(module).__name__:
                # 1.4 workaround: now there's an original_name member so just use that
                if hasattr(module, 'original_name'):
                    is_script_conv = 'Conv' in module.original_name
                # 1.3 workaround: check if this has the same constants as a conv module
                else:
                    is_script_conv = (
                            all_in(module.__dict__['_constants_set'], conv_constants)
                            and all_in(conv_constants, module.__dict__['_constants_set']))

            is_conv_layer = isinstance(module, nn.Conv2d) or is_script_conv

            if is_conv_layer and module not in self.backbone.backbone_modules:
                nn.init.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    if cfg.use_focal_loss and 'conf_layer' in name:
                        if not cfg.use_sigmoid_focal_loss:
                            module.bias.data[0] = np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                            module.bias.data[1:] = -np.log(module.bias.size(0) - 1)
                        else:
                            module.bias.data[0] = -np.log(cfg.focal_loss_init_pi / (1 - cfg.focal_loss_init_pi))
                            module.bias.data[1:] = -np.log((1 - cfg.focal_loss_init_pi) / cfg.focal_loss_init_pi)
                    else:
                        module.bias.data.zero_()
                        
    def train(self, mode=True):
        super().train(mode)

        if cfg.freeze_bn:
            self.freeze_bn()

    def freeze_bn(self, enable=False):
        """ Adapted from https://discuss.pytorch.org/t/how-to-train-with-frozen-batchnorm/12106/8 """
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

        for module in self.modules():
            if isinstance(module, nn.BatchNorm1d):
                module.train() if enable else module.eval()

                module.weight.requires_grad = enable
                module.bias.requires_grad = enable

    def mask_generator(self, word_id):
        word_id_tmp = word_id.cpu()
        word_mask = []
        for i in range(len(word_id_tmp)):
            tmp_word = word_id_tmp[i]
            tmp_mask = tmp_word[tmp_word>0]
            tmp_mask = list(np.ones_like(tmp_mask))
            tmp_mask += [0] * (len(tmp_word) - len(tmp_mask))
            word_mask.append(tmp_mask)
            assert(len(tmp_mask) == len(tmp_word))
        # word_mask = torch.tensor(word_mask, dtype=torch.float64).cuda()
        word_mask = torch.LongTensor(word_mask).cuda()
        return word_mask
    
    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear') + y

    def semantics_retrieval(self, cmp_vis_feat, lang_emb):
        B, C, H, W = cmp_vis_feat.size()
        vis_seq = cmp_vis_feat.reshape(B, C, H*W).permute([0,2,1]) #Querying [B, HW, C]
        lang_key = lang_emb.permute([0, 2, 1]) #[B, C, L]
        lang_val = lang_emb #[B, L, C]
        affinity_matrix = F.softmax((vis_seq@lang_key)*C**-.5, dim = -1) # [B, HW, L]
        lang_kernel = affinity_matrix@lang_val #[B, HW, C]
        return lang_kernel

    def encoder(self, x, word_id):
        _, _, img_h, img_w = x.size()
        cfg._tmp_img_h = img_h
        cfg._tmp_img_w = img_w

        #----------word mask----------
        word_mask = self.mask_generator(word_id)
        embedded = self.BertEmbededing(word_id)
        extended_attention_mask = word_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #----------lang encoding--------------
        for layer_module in self.layers[0:12]:
            embedded = layer_module(embedded, attention_mask = extended_attention_mask)

        x = self.img_bn(x)
        c0 = self.layer0(x)
        #----------layer1 interaction---------------------
        c1 = self.layer1(c0) #[24, 256, 80, 80]
        c1_side = self.compression_one(c1) # [24, 256, 10, 10]
        c1_enh, lang = self.down_vision_layer1(c1), self.down_lang_layer1(embedded)
        lang_kernel = self.semantics_retrieval(c1_side, lang)
        c1_enh = self.dynamic_convolution1(c1_enh, lang_kernel)
        c1 = F.relu(self.bn1(c1 + self.up_vision_layer1(c1_enh)))
        #----------layer2 interaction---------------------
        c2 = self.layer2(c1) #[24, 512, 40, 40]
        c2_side = self.compression_two(c2) # [24, 256, 10, 10]
        c2_enh, lang = self.down_vision_layer2(c2), self.down_lang_layer2(embedded)
        lang_kernel = self.semantics_retrieval(c2_side, lang)
        c2_enh = self.dynamic_convolution2(c2_enh, lang_kernel)
        c2 = F.relu(self.bn2(c2 + self.up_vision_layer2(c2_enh)))
        #----------layer3 interaction---------------------
        c3 = self.layer3(c2)
        c3_side = self.compression_three(c3) # [24, 256, 10, 10]
        c3_enh, lang = self.down_vision_layer3(c3), self.down_lang_layer3(embedded)
        lang_kernel = self.semantics_retrieval(c3_side, lang)
        c3_enh = self.dynamic_convolution3(c3_enh, lang_kernel)
        c3 = F.relu(self.bn3(c3 + self.up_vision_layer3(c3_enh)))
        #----------layer4 interaction---------------------
        c4 = self.layer4(c3)
        c4_side = self.compression_four(c4) # [24, 256, 10, 10]
        c4_enh, lang = self.down_vision_layer4(c4), self.down_lang_layer4(embedded)
        lang_kernel = self.semantics_retrieval(c4_side, lang)
        c4_enh = self.dynamic_convolution4(c4_enh, lang_kernel)
        c4 = F.relu(self.bn4(c4 + self.up_vision_layer4(c4_enh)))
        return c0, c1, c2, c3, c4, embedded

    def forward(self, x, word_id):
        B, C, H, W = x.size()
        c0, c1, c2, c3, c4, l4 = self.encoder(x, word_id)
        output = self.decoder(x, c0, c1, c2, c3, c4, l4)
        ############################################################################################################
        if self.training:
            return output, output, output, output, output
        else:
            return output

import torch
import torch.nn as nn
from timm.models.resnet import Bottleneck

from model import get_model
from model import MODEL

import math
from util.util import add_jitter
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange
from timm.models.layers import trunc_normal_



class DWConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,padding=1,dilation=1):
        super().__init__()
        self.dconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,padding=padding,stride=stride, groups=in_channels,dilation=1),
            nn.InstanceNorm2d(in_channels),nn.SiLU())
        self.pconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),nn.SiLU())
 
    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)





class Mlp(nn.Module):
    def __init__(self, 
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.SiLU,
                 drop=0.):
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
        x = self.fc2(x)
        x = self.drop(x)

        return x

class Attention(nn.Module):
    def __init__(self,q,k,v,norm,act,pe=None):
        super().__init__()
        self.q=q
        self.k=k
        self.v=v
        self.scale = self.q.size(-1)**-0.5
        b,h,n,c = self.q.shape
        
        self.ffn = Mlp(h*c)
    
        self.pe = pe
    def forward(self):
        # qkv shape b h N c ::pe b h N n
        b,h,n,c = self.q.shape
        self.k = self.k.transpose(-1,-2)# b h c n
        attn_score = self.pe + ((self.q @ self.k)*self.scale).softmax(-1) # b h N n
        atten = attn_score @ self.v # b h N c
        out = atten.permute(0,2,1,3).reshape(b,n,-1) # b N h*c
        out = self.ffn(out) # b N h*c        
        return out.contiguous() # b N h*c

   
class ToProxy2(nn.Module):
    def __init__(self,dim=512,head_num = 8,agent_q_num=8**2,agent_kv_num=8**2,norm=nn.InstanceNorm2d,act=nn.SiLU):
        super().__init__()
        self.agent_q_num = agent_q_num
        self.learnable_q = nn.Parameter(torch.zeros(1,agent_q_num,dim))
        self.learnable_kv = nn.Parameter(torch.zeros(1,agent_kv_num,dim))
        self.q = nn.Linear(dim,dim)
        self.kv_x = nn.Linear(dim,dim*2)
        self.kv_y = nn.Linear(dim,dim*2)
        self.norm = norm(dim)
        self.act = act()
        self.head = head_num
        self.q_2 = nn.Linear(dim,dim)
        self.pe1 = nn.Parameter(torch.zeros(1,head_num,1,agent_kv_num))
        self.pe2 = nn.Parameter(torch.zeros(1,head_num,agent_q_num,1))
        
    def forward(self,x):
        b,h,w,c = x.shape
        head = self.head
         
         
         
        learnable_q = self.learnable_q.repeat(b,1,1) # b N(agent_q_num) c
        q = self.q(learnable_q).reshape(b,self.agent_q_num,head,c//head).permute(0,2,1,3) # b h N c
        k,v = self.kv_x(x).chunk(2,dim=-1) # b n c 
        
        k = k.reshape(b,h*w,head,c//head).permute(0,2,1,3) # b h n c
        v = v.reshape(b,h*w,head,c//head).permute(0,2,1,3) # b h n c
        pe2 = self.pe2.repeat(b,1,1,h*w)
        atn = Attention(q,k,v,self.norm,self.act,pe=pe2).to(x.device) 
        fin_out = atn() 
        h1 = w1 = int(self.agent_q_num**0.5)
        out = fin_out.reshape(b,h1,w1,-1).permute(0,3,1,2) # b c h1 w1
        out = F.interpolate(out,(h,w)).permute(0,2,3,1) #b h w c 
        out = out + x
        mskip = out
        # 
        q = self.q_2(out).reshape(b,h*w,-1).reshape(b,h*w,head,c//head).permute(0,2,1,3) # b h N c
        learnable_v = self.learnable_kv.repeat(b,1,1) # b N1(agent_kv_num) c
        pe1 = self.pe1.repeat(b,1,h*w,1)
        b,N1,c = learnable_v.shape
        k,v = self.kv_y(learnable_v).chunk(2,dim=-1)
        k = k.reshape(b,N1,head,c//head).permute(0,2,1,3) # b h n c
        v = v.reshape(b,N1,head,c//head).permute(0,2,1,3) # b h n c
        atn = Attention(q,k,v,self.norm,self.act,pe=pe1).to(x.device)
        out = atn().reshape(b,h,w,-1)+mskip+x
        
        return out.contiguous()



class OmniBlock(nn.Module):
    def __init__(self,hidden_dim: int = 0):
        
        super().__init__()
        self.toProxy = ToProxy2(dim=hidden_dim)
        self.conv33 = DWConv(hidden_dim, hidden_dim, 3,padding=1)
        self.conv55 = DWConv(hidden_dim, hidden_dim, 5,padding=2)
        self.mlp = nn.Sequential(nn.Conv2d(hidden_dim*2, hidden_dim, 1),nn.InstanceNorm2d(hidden_dim),nn.SiLU())  
        self.l = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.InstanceNorm2d(hidden_dim)
        self.act = nn.SiLU()
        self.l2 = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        conv_input = x.permute(0, 3, 1, 2)
        out = self.toProxy(x)
        y33 = self.conv33(conv_input).permute(0,2,3,1) # b h w c 
        y55 = self.conv55(conv_input).permute(0,2,3,1) # b h w c
        y33 = self.mlp(torch.cat([y33,y55],dim=-1).permute(0,3,1,2)).permute(0,2,3,1)
        
        out = self.act(self.norm(self.l(out+y33).permute(0,3,1,2))).permute(0,2,3,1)
        out = self.l2(out)
        out = out + x
        
        return out.contiguous()
    


class OmniBlockWrapper(nn.Module):
    def __init__(self,hidden_dim: int = 0):
            super().__init__()
            self.blocks = nn.ModuleList([OmniBlock(hidden_dim=hidden_dim) for _ in range(1)])
    def forward(self, x):
        # x -  B H W C
        out = x
        for block in self.blocks:
            out = block(out)
        return out
    


def conv3x3(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def deconv2x2(in_planes, out_planes, stride = 1, groups = 1, dilation = 1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=2, stride=stride, groups=groups, bias=False, dilation=dilation)

class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim*2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        x= self.norm(x)
        return x



class DecoderStage(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            norm_layer=nn.LayerNorm,
            upsample=None,
            use_checkpoint=False,
    ):
        super().__init__()
        self.dim = dim
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([OmniBlockWrapper(hidden_dim=dim) for i in range(depth)])

        if True:  
            def _init_weights(module: nn.Module):
                for name, p in module.named_parameters():
                    if name in ["out_proj.weight"]:
                        p = p.clone().detach_()  
                        nn.init.kaiming_uniform_(p, a=math.sqrt(5))

            self.apply(_init_weights)

        if upsample is not None:
            self.upsample = upsample(dim=dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x
    

class Decoder(nn.Module):
    def __init__(self, dims_decoder=[512, 256, 128, 64], depths_decoder=[3, 9, 9, 7], drop_path_rate=0.2,
                 norm_layer = nn.LayerNorm):
        super().__init__()
        dpr_decoder = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths_decoder))][::-1]
        self.layers_up = nn.ModuleList()
        for i_layer in range(len(depths_decoder)):
            layer = DecoderStage(
                dim=dims_decoder[i_layer],
                depth=depths_decoder[i_layer],
                norm_layer=norm_layer,
                upsample=PatchExpand2D if (i_layer != 0) else None,
            )
            self.layers_up.append(layer)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward(self, x):
        x = rearrange(x,'b c h w -> b h w c')
        out_features = []
        for i, layer in enumerate(self.layers_up):
            x = layer(x)
            if i != 0:
                out_features.insert(0, rearrange(x,'b h w c -> b c h w'))
        return out_features

class Neck(nn.Module):
    def __init__(self, block, layers, width_per_group = 64, norm_layer = None, ):
        super(Neck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.base_width = width_per_group
        self.inplanes = 64 * block.expansion
        self.dilation = 1
        self.bn_layer = self._make_layer(block, 128, layers, stride=2)

        self.conv1 = conv3x3(16 * block.expansion, 32 * block.expansion, 2)
        self.bn1 = norm_layer(32 * block.expansion)
        self.conv2 = conv3x3(32 * block.expansion, 64 * block.expansion, 2)
        self.bn2 = norm_layer(64 * block.expansion)
        self.conv21 = nn.Conv2d(32 * block.expansion, 32 * block.expansion, 1)
        self.bn21 = norm_layer(32 * block.expansion)
        self.conv31 = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bn31 = norm_layer(64 * block.expansion)
        self.convf = nn.Conv2d(64 * block.expansion, 64 * block.expansion, 1)
        self.bnf = norm_layer(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride = 1, dilate = False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(conv1x1(self.inplanes, planes * block.expansion, stride),
                                       norm_layer(planes * block.expansion), )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, base_width=self.base_width, dilation=previous_dilation, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, base_width=self.base_width, dilation=self.dilation, norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def forward(self, x):
        fpn0 = self.relu(self.bn1(self.conv1(x[0])))
        fpn1 = self.relu(self.bn21(self.conv21(x[1]))) + fpn0
        sv_features = self.relu(self.bn2(self.conv2(fpn1))) + self.relu(self.bn31(self.conv31(x[2])))
        sv_features = self.relu(self.bnf(self.convf(sv_features)))
        sv_features = self.bn_layer(sv_features)

        return sv_features.contiguous()

class OmniAD(nn.Module):
    def __init__(self, model_t, model_s):
        super(OmniAD, self).__init__()
        self.net_t = get_model(model_t)
        self.Neck = Neck(Bottleneck, 3)
        self.net_s = Decoder(depths_decoder=model_s['depths_decoder'])

        self.frozen_layers = ['net_t']
        
    def freeze_layer(self, module):
        module.eval()
        for param in module.parameters():
            param.requires_grad = False

    def train(self, mode=True):
        self.training = mode
        for mname, module in self.named_children():
            if mname in self.frozen_layers:
                self.freeze_layer(module)
            else:
                module.train(mode)
        return self

    def forward(self, imgs):
        feats_t = self.net_t(imgs)
        feats_t = [f.detach() for f in feats_t]
        oce_out = self.Neck(feats_t)  # 16 512 8 8
        b,c,h,w = oce_out.shape
        scale = 20 if self.training else 0
        oce_out = add_jitter(oce_out.reshape(b,c,h*w).permute(0,2,1),scale=scale).reshape(b,h,w,c).permute(0,3,1,2).contiguous()
        feats_s = self.net_s(oce_out)
        return feats_t, feats_s

@MODEL.register_module
def omniAD(pretrained=False, **kwargs):
    model = OmniAD(**kwargs)
    return model

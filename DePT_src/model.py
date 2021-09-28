from .utils import *
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EXP_CONFIG_ID = 1

def getArgs():
    args = {

            'lr':               [0.005, 0.005, 0.01],
            'Tmax':             10,  # perception field
            'ddqn_freq':        40,  # only used when teacher is cacot
            'batch_size':       [64, 128, 256][EXP_CONFIG_ID],
            'n_epochs':         [100, 100, 400 ][EXP_CONFIG_ID],

            'which_teacher':    ['cacot','colight'][1],
            'colight_fname':    'DePT_src/round_60_inter_0.h5',
            'ablation1_cone':   False   ,
            'ablation2_time':   False   ,
            'only_1cone':       False   ,

            'dim_feat':         12,
            'use_pretrained_cone':    True,
            'cacot_fname':      'DePT_src/cacot_66.pth',
            'cacot_bar_fname':  'DePT_src/cacot_66_bar.pth',
            'n_blocks':         [6, 12, 12][EXP_CONFIG_ID],
            'n_head':           [8, 8,  6][EXP_CONFIG_ID],
            'dim_k':            [16, 32, 32][EXP_CONFIG_ID],
            'dim_v':            [16, 32, 32][EXP_CONFIG_ID],

            'output_hidden':    [16, 32, 32][EXP_CONFIG_ID],
            'dim_ffn':          [24, 32, 32][EXP_CONFIG_ID],
            'dim_model':        12,
            'scale_emb':        [1.0, None][1],  # used in partial CAV (partial observation) case
            'n_actions':        4,

            'dropout':          0.2,
            'use_attn_residual':   True,

            # unused
            'dim_pe':           None,
            'dim_te':           None,

            # decayFun
            'N_decayer_caus':   20,
            'decayer_ini_caus': [-10, 10],
            'N_decayer_time':   20,
            'decayer_ini_time': [5, 500],       # need testing
            'decay_k_ini':      0.1,            # 0.1 is good, 
            'multiGPU':         [False, [0,1,2,3]] [0] ,

            }
    args['dims_flowSpeed'] = [args['dim_feat'], 20, 20  ,1]   # MLP layers for vFunc
    args['dim_model'] = args['dim_feat']

    return args




class DelayedPropagationTransformer(nn.Module):
    def __init__(self, ID2Pos, N_nodes):
        self.ID2Pos = torch.tensor(ID2Pos, device=DEVICE, requires_grad=False).float()  # tensor, shape: [N_nodes, 2]
        self.N_nodes = N_nodes
        args = getArgs()
        self.args_cacot = args
        args['N_nodes'] = N_nodes
        self.init(args,**args)

    def init(self, args, dim_feat=None, n_blocks=None, n_head=None, dim_k=None, dim_v=None, dim_model=None, dim_ffn=None, dims_flowSpeed=None, dim_pe=None, dim_te=None, scale_emb=None, dropout=None, use_attn_residual=None, n_actions=None, output_hidden=None, Tmax=None, ablation1_cone=None, only_1cone=None, **w):
        super().__init__()
        self.n_actions = n_actions
        self.output_hidden = output_hidden
        self.tokens_st_list = []
        self.only_1cone = only_1cone
        self.n_head = n_head

        self.Tmax = Tmax
        self.dropout = nn.Dropout(p=dropout)
        self.layer_stack = nn.ModuleList([
            EncoderLayer(dim_model, dim_ffn, n_head, dim_k, dim_v, dropout, use_attn_residual)
            for _ in range(n_blocks)])
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)
        if scale_emb: 
            scale_emb = torch.tensor(scale_emb, device=DEVICE).float() # should be a scalar
            self.scale_emb = nn.Parameter(scale_emb, requires_grad=True)
            self.scale_emb.requires_grad = True
        else:
            self.scale_emb = None


        self.dim_model = dim_model
        # self.cone = CausalityCone(args, **args)
        self.cone = repeatedModules(CausalityCone, [args], 1 if only_1cone else n_blocks, 1, True)

        # self.final_mlp = getMLP([self.dim_model, self.output_hidden, self.n_actions]).to(DEVICE)
        self.final_mlp = getMLP([self.dim_model, self.n_actions]).to(DEVICE)



    def forward_tokens(self, tokens):
        # tokens shape: [batch_size, num_tokens, 1(nodeID)+1(timeFineGrained)+dim_feat]
        # returned shapes:
        #     x: [batch_size, num_tokens, dim_model]
        #     attn_cone: [batch_size, n_head(simply expand), num_tokens, num_tokens]
            
        attn_ori_list = []

        nodeID = tokens[...,0].long()
        nodePos = self.ID2Pos[nodeID] 
        timeFineGrained = tokens[...,1]

        src_mask = gen_mask(timeFineGrained)

        # pe = self.position_enc(nodeID)
        # te = self.time_enc(time_quantize(timeFineGrained))
        timeCoarseGrained = time_quantize(timeFineGrained)


        x = tokens[...,2:-1]
        actionIdx = (tokens[...,-1]-1).long()
        if self.scale_emb is not None:
            x *= self.scale_emb
        x = self.dropout(x)
        x = self.layer_norm(x)

        for il, enc_layer in enumerate(self.layer_stack):

            if self.only_1cone:
                attn_cone = self.cone[0](x, nodeID, nodePos, timeFineGrained, timeCoarseGrained)
                # [batch_size, n_head, num_tokens, num_tokens]
            else:
                attn_cone = self.cone[il](x, nodeID, nodePos, timeFineGrained, timeCoarseGrained)

            x, attn_ori = enc_layer(x, attn_cone, attn_mask=src_mask) # mask -wz
            attn_ori_list.append(attn_ori)

        return x, attn_ori_list, attn_cone
        
    def predict(self, inputs):
        # nodes:        [N_nodes, 1(nodeID)+1(timeFineGrained)+dim_feat]
        # tokens_st_tensor:    [batch_size, Tmax, N_nodes, 1(nodeID)+1(timeFineGrained)+dim_feat]
        # tokens_cur_out:       [batch_size, N_nodes, dim_model]
        # output: action:     [batch_size, N_nodes, n_actions]
        # if count==0:
        #     del self.tokens_st_list
        #     self.tokens_st_list = []
        #     for t in range(self.Tmax):
        #         empty = torch.zeros(*nodes.shape, device=DEVICE, requires_grad=False)
        #         empty[:,0] = nodes[:,0]
        #         empty[:,1] = nodes[:,1]-t
        #         self.tokens_st_list.append(empty)
        # del self.tokens_st_list[-1]
        # self.tokens_st_list.insert(0, nodes)
        # tokens_st_tensor = torch.stack(self.tokens_st_list, axis=0) # [Tmax, N_nodes, N]

        tokens_st_tensor, _ = inputs
        tokens_st_tensor = torch.tensor(tokens_st_tensor,device=DEVICE, dtype=torch.float32)
        batch_size, Tmax, N_nodes, feat_ = tokens_st_tensor.shape
        tokens = tokens_st_tensor.view(batch_size, Tmax*N_nodes, feat_)
        tokens_out, attn_ori_list, attn_cone = self.forward_tokens(tokens)
        tokens_st_out = tokens_out.view(batch_size, Tmax, N_nodes, self.dim_model)
        tokens_cur_out = tokens_st_out[:,0,...]
        action = self.final_mlp(tokens_cur_out)
        attn = [attn_ori_list, attn_cone]
        return action, attn


    def forward(self,x):
        y, attn = self.predict([x,None])
        return y



def load_cone(cones, which_roadmap, n_head=None):
    def load_1cone(cone, which_roadmap, which_block=0, which_head=0):
        subnet = cone.coneDecay[which_head].decayFun.mlp
        load_model(subnet, f'DePT_src/pre_m1 @ {which_roadmap} $ {which_block} # {which_head}')
        subnet = cone.timeDecay[which_head].mlp 
        load_model(subnet, f'DePT_src/pre_m2 @ {which_roadmap} $ {which_block} # {which_head}')
        subnet =  cone.coneDecay[which_head].vFunc_o
        load_model(subnet, f'DePT_src/pre_v1 @ {which_roadmap} $ {which_block} # {which_head}')
        subnet =  cone.coneDecay[which_head].vFunc_d
        load_model(subnet, f'DePT_src/pre_v2 @ {which_roadmap} $ {which_block} # {which_head}')
        subnet = cone.coneDecay[which_head].speedStLUT

        # value = subnet.state_dict()['stLUT']*0 + 2
        value = subnet.state_dict()['stLUT']*0 + 3*   15

        value = value + torch.randn(*value.shape,device=value.device)*0.05
        subnet.load_state_dict(OrderedDict({'stLUT': value}))
        subnet = cone.attnStLUT[which_head]
        value = subnet.state_dict()['stLUT']*0
        value = value + torch.randn(*value.shape,device=value.device)*0.05
        subnet.load_state_dict(OrderedDict({'stLUT': value}))

    only_1cone = (len(cones)==1)
    if only_1cone:
        load_1cone(cone, which_roadmap)
        return
    for il, cone in enumerate(cones):
        for ih in range(n_head):
            load_1cone(cone, which_roadmap, il, ih)
    return





class coneDecay(nn.Module):
    def __init__(self, N_nodes, dims_flowSpeed, N_decayer_caus, decayer_ini_caus, decay_k_ini, **w):

        super().__init__()
        self.speedStLUT = STLUT(N_nodes)

        self.vFunc_o = getMLP(dims_flowSpeed, prep_str='/100').to(DEVICE)
        self.vFunc_d = getMLP(dims_flowSpeed, prep_str='/100').to(DEVICE)

        # self.decayFun = DecayFun(N_decayer_caus, decayer_ini_caus, decay_k_ini)
        self.decayFun = getMLP_11([1,20,20,1], prep_str='/1e4')



    def forward(self,nodeID2,timeID2,features,posDiff,timeDiff):
        # nodePos:          [batch_size, num_tokens, 2]
        # timeFineGrained:  [batch_size, num_tokens]
        # features:         [batch_size, num_tokens, dim_feat]
        # return attn_cone: [batch_size, n_head, num_tokens, num_tokens]
        # return:    [batch_size, num_tokens, num_tokens]
        speed_lut = self.speedStLUT(nodeID2, timeID2)
        speed_o = self.vFunc_o(features)
        speed_d = self.vFunc_d(features)
        speed_d = speed_d.transpose(1,2)
        speed_btt = (speed_o + speed_d + speed_lut)/3
        epsilon = timeDiff*speed_btt - posDiff
        res = self.decayFun(epsilon)
        return res






class EncoderLayer(nn.Module):
    def __init__(self, hidden_dim=None, dim_inner=None, n_head=None, dim_k=None, dim_v=None, dropout=None, use_attn_residual=None,**w):
        # hidden_dim = dim_model
        super(EncoderLayer, self).__init__()
        self.multi_attn = CausalityConeAttention(n_head, hidden_dim, hidden_dim, dim_k, dim_v, dropout, use_attn_residual)
        self.pos_ffn = PositionwiseFeedForward(hidden_dim, dim_inner, dropout=dropout)

    def forward(self, enc_input, attn_cone, attn_mask=None):
        enc_output, attn_ori = self.multi_attn(enc_input, attn_cone, mask=attn_mask)
        enc_output = self.pos_ffn(enc_output)
        return enc_output, attn_ori





class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=None):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        # self.cone = cone

    def forward(self, q, k, v, attn_cone, mask=None):
        # k,q,v: [batch_size, n_head, seq_len, dim_feat]

        attn_ori = torch.matmul(q / self.temperature, k.transpose(2, 3)) # b x l x l
        attn = attn_ori + attn_cone
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)
        return output, attn_ori

class CausalityConeAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, dim_in, dim_out, dim_k, dim_v, dropout, use_attn_residual):
        # dim_in = dim_out = dim_model
        super().__init__()

        self.n_head = n_head
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.use_attn_residual = use_attn_residual

        self.Wq = nn.Linear(dim_in, n_head * dim_k, bias=False)
        self.Wk = nn.Linear(dim_in, n_head * dim_k, bias=False)
        self.Wv = nn.Linear(dim_in, n_head * dim_v, bias=False)
        self.fc = nn.Linear(n_head * dim_v, dim_out, bias=False)

        self.attention = ScaledDotProductAttention(temperature=dim_k ** 0.5, attn_dropout=dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_out, eps=1e-6)


    def forward(self, x, attn_cone, mask=None):
        q = k = v = x  # shape: [batch_size=1, seq_len, dim_feat]


        dim_k, dim_v, n_head = self.dim_k, self.dim_v, self.n_head
        batch_size, len_q, len_k, len_v = q.shape[0], q.shape[1], k.shape[1], v.shape[1]

        residual = v
        # Pass through the pre-attention projection: b x l x (h*d)
        # Separate different heads: b x l x h x d
        q = self.Wq(q).view(batch_size, len_q, n_head, dim_k)
        k = self.Wk(k).view(batch_size, len_k, n_head, dim_k)
        v = self.Wv(v).view(batch_size, len_v, n_head, dim_v)

        # Transpose for attention dot product: b x h x l x d
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        v_out, attn_ori = self.attention(q, k, v, attn_cone, mask=mask)

        # Transpose to move the head dimension back: b x l x n x d
        # Combine the last two dimensions to concatenate all the heads together: b x l x (n*d)
        v_out = v_out.transpose(1, 2).reshape(batch_size, len_q, -1)
        v_out = self.dropout(self.fc(v_out))

        if self.use_attn_residual:
            v_out += residual

        v_out = self.layer_norm(v_out)

        return v_out, attn_ori


class PositionalEncoding(nn.Module):

    def __init__(self, dim_hidden, n_position=200):
        super(PositionalEncoding, self).__init__()
        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, dim_hidden))

    def _get_sinusoid_encoding_table(self, n_position, n_dim):
        ''' Sinusoid position encoding table '''

        js = torch.arange(dim, dtype=torch.int32)
        js = torch.pow(10000, 2 * (js // 2) / n_dim) 

        pos = torch.arange(n_position, dtype=torch.float32)
        sinusoid_table = pos[..., None] / js[None, ...]
        sinusoid_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        return sinusoid_table.unsqueeze(0) # [1, n_pos, n_dim]

    def forward(self, x):
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, dim_in, dim_hidden, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(dim_in, dim_hidden) # position-wise
        self.w_2 = nn.Linear(dim_hidden, dim_in) # position-wise
        self.layer_norm = nn.LayerNorm(dim_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class DecayFun(nn.Module):
    def __init__(self, N_decayer, ini_range, decay_k_ini, **w):
        super().__init__()
        

        decayers = torch.linspace(ini_range[0],ini_range[1],N_decayer, device=DEVICE)
        ks = torch.ones(N_decayer, device=DEVICE)*decay_k_ini
        betas = torch.ones(N_decayer, device=DEVICE)*2

        self.N_decayer = N_decayer
        self.decayers = nn.Parameter(decayers, requires_grad=True)
        self.ks = nn.Parameter(ks, requires_grad=True)
        self.betas = nn.Parameter(betas, requires_grad=False)


    def forward(self, x):
        res = x*0
        for i in range(self.N_decayer):
            d = self.decayers[i]
            k = self.ks[i]
            beta = self.betas[i]
            if d>=0:
                res -= F.softplus(x-d, beta.item())*k
            else:
                res -= F.softplus(d-x, beta.item())*k
        return res







class STLUT(nn.Module):
    def __init__(self, N_nodes):
        super().__init__()

        # stLUT = torch.randn(N_nodes, 24, N_nodes, 24)
        stLUT = torch.randn(N_nodes, 1, N_nodes, 1)

        self.stLUT = nn.Parameter(stLUT, requires_grad=True)
    def forward(self, nodeID2, timeID2):
        res = self.stLUT[nodeID2[...,0],timeID2[...,0],nodeID2[...,1],timeID2[...,1]]
        # res = self.stLUT[nodeID2[...,0],0,nodeID2[...,1],0]
        return res


def repeatedModules(mod, Hargs, n_repeat, n_positioning, has_star):
    # Hargs: a list, if has_star, the first one must be the dict
    # n_positioning: number of positioning parameters of the mod function
    # has_star: whether to append a **Hargs[0] at the end

    mods = nn.ModuleList([])
    for i in range(n_repeat):
        if has_star: 
            modi = mod(*Hargs[:n_positioning], **Hargs[0])
        else:
            modi = mod(*Hargs[:n_positioning])
        mods.append(modi)
    return mods

class CausalityCone(nn.Module):
    def __init__(self, args, dims_flowSpeed, N_nodes, n_head, N_decayer_caus, decayer_ini_caus, N_decayer_time, decayer_ini_time, decay_k_ini, ablation1_cone, ablation2_time, only_1cone, **w):
        super().__init__()
        self.n_head = n_head
        self.only_1cone = only_1cone
        self.ablation1_cone = ablation1_cone
        self.ablation2_time = ablation2_time
        # if only_1cone:
        #     self.coneDecay = coneDecay(**args)
        #     self.timeDecay = getMLP_11([1,20,20,1])
        #     self.attnStLUT = STLUT(N_nodes)
        # else:
        self.coneDecay = repeatedModules(coneDecay, [args], 1 if only_1cone else n_head, 0, True)
        self.timeDecay = repeatedModules(getMLP_11, [ [1,20,20,1] ],  1 if only_1cone else n_head, 1, False)
        self.attnStLUT = repeatedModules(STLUT, [N_nodes],  1 if only_1cone else n_head, 1, False)



    def forward(self, features, nodeID, nodePos, timeFineGrained, timeCoarseGrained):
        timeDiff = pairSub_batch(timeFineGrained, timeFineGrained)   # [batch_size, num_tokens, num_tokens]
        posDiff = pairSub_batch(nodePos, nodePos).norm(p=1,dim=-1)   # [batch_size, num_tokens, num_tokens]
        nodeID2 = idx2D_batch(nodeID)
        timeID2 = idx2D_batch(timeCoarseGrained)

        v0 = torch.zeros(*posDiff.shape, device=DEVICE, dtype=torch.float32)
        if self.only_1cone:    
            v1 = self.coneDecay[0](nodeID2,timeID2,features,posDiff,timeDiff) if not self.ablation1_cone else v0
            v2 = self.attnStLUT[0](nodeID2, timeID2) + self.timeDecay[0](timeDiff) if not self.ablation2_time else v0
            btt = v1 + v2
            batch_size, num_tokens = timeFineGrained.shape
            bhtt = btt.unsqueeze(1).expand(batch_size, self.n_head, num_tokens, num_tokens)
            return bhtt
        bhtt = []
        for ih in range(self.n_head):
            v1 = self.coneDecay[ih](nodeID2,timeID2,features,posDiff,timeDiff) if not self.ablation1_cone else v0
            v2 = self.attnStLUT[ih](nodeID2, timeID2) + self.timeDecay[ih](timeDiff) if not self.ablation2_time else v0
            btt = v1 + v2
            bhtt.append(btt)

        bhtt = torch.stack(bhtt).transpose(0,1)

        return bhtt







def pairSub(x, y):
    # input and output are all torch.Tensor
    z = x.unsqueeze(1) - y.unsqueeze(0)
    return z

def pairSub_batch(x, y):
    # input and output are all torch.Tensor
    # output index: z[ib, i,j] = x[ib, i] - y[ib, j]
    z = x.unsqueeze(2) - y.unsqueeze(1)
    return z


def pairCat(x, y):
    # input and output are all torch.Tensor
    # output index: z[i,j] = [x[i], y[j]]
    N, dim_feat = x.shape
    x = x.unsqueeze(1).expand(N,N,dim_feat)
    y = y.unsqueeze(0).expand(N,N,dim_feat)
    z = torch.cat([x,y], dim=-1)
    return z

def idx2D_batch(x0):
    # input and output are all torch.Tensor
    batch_size, N = x0.shape
    x = x0.unsqueeze(2).expand(batch_size,N,N)
    y = x0.unsqueeze(1).expand(batch_size,N,N)
    z = torch.stack([x,y], dim=-1)
    return z



def time_quantize(timeFineGrained):
    # input/output have same shape, both are tensor
    # output dtype is int
    # quantize input (in seconds) into hourly based index
    # 24 hours = 24*3600 = 86400 s
    has_dynamic_flow = 0
    if has_dynamic_flow:
        timeCoarseGrained = timeFineGrained/3600
        timeCoarseGrained = timeCoarseGrained.long()
    else:
        timeCoarseGrained = torch.zeros(*timeFineGrained.shape).long()

    return timeCoarseGrained


def gen_mask(timeFineGrained, want_mask=True):
    # return:           [batch_size, n_head=1, num_tokens, num_tokens]
    if want_mask:
        btt = pairSub_batch(timeFineGrained,timeFineGrained)
        btt = (btt<0).byte()   # v==1 elements will be musked
        return btt
    else:
        return None



class DoNothing:
    def __call__(self, *a):
        return 0.

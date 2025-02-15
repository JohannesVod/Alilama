import torch.nn as nn
import torch.nn.functional as F
import torch
from TRAINCONFIG import *
from typing import Optional

from dataclasses import dataclass, field

@dataclass
class ModelParameters:
    model_name:str = "alilama"
    d_model:int = 128
    blocks:int = 8
    max_seq_len:int = 128
    num_heads:int = 8
    hidden_dim:int = 4*d_model
    head_width:int = d_model//num_heads
    weight_tying_dict: dict = field(default_factory=dict)

# See ALIBI: https://arxiv.org/pdf/2108.12409.pdf
def get_relative_positions(seq_len: int) -> torch.tensor:
    matrix = torch.zeros((seq_len, seq_len), dtype=torch.int64)
    for i in range(seq_len):
        matrix[i] = torch.arange(-i, seq_len-i)
    return matrix

def get_alibi_slope(num_heads):
    step_size = 8/num_heads
    arr = [2**(-i*step_size) for i in range(1, num_heads+1)]
    return torch.tensor(arr).unsqueeze(-1).unsqueeze(-1)

# Multiheaded self-attention
class MHSA(nn.Module):
    def __init__(self, device, params):
        super(MHSA, self).__init__()
        self.params = params
        # query/key/value projections
        self.q = nn.Linear(params.d_model, params.d_model, bias=False)
        self.k = nn.Linear(params.d_model, params.d_model, bias=False)
        self.v = nn.Linear(params.d_model, params.d_model, bias=False)
        # From ALIBI paper. We dont want to train this parameter:
        self.m = get_alibi_slope(params.num_heads).to(device)
        self.pos_embed = get_relative_positions(params.max_seq_len).to(device)
        # final projection before the residual connection:
        self.preadditive = nn.Linear(params.d_model, params.d_model, bias=False)
        self.att_matr = None # TODO: Remove

    def weightTye(self, other):
        self.q.weight = other.q.weight
        self.k.weight = other.k.weight
        self.v.weight = other.v.weight
        self.preadditive.weight = other.preadditive.weight

    def forward(self, x, mask):
        N, t, d = x.shape # input shape
        
        def SplitIntoHeads(tens):
            # Reshapes to (N, num_heads, t, head_width)
            reshaped = tens.reshape((N, t, self.params.num_heads, self.params.head_width)) 
            return reshaped.transpose(1, 2)

        q = SplitIntoHeads(self.q(x)) # (N, num_heads, t, head_width)
        k = SplitIntoHeads(self.k(x)) # (N, num_heads, t, head_width)
        v = SplitIntoHeads(self.v(x)) # (N, num_heads, t, head_width)

        # compute attention:
        att = q@k.transpose(-2, -1) # (N, num_heads, t, t)
        att = att/self.params.head_width**(1/2) # rescale
        # apply relative embedding (ALIBI):
        att += (self.m*self.pos_embed[:t, :t]).unsqueeze(0)
        att = att + mask[:, :, :t, :t]
        att = F.softmax(att, -1)
        self.att_matr = att
        res = att @ v
        res = res.transpose(1, 2) # (N, t, num_heads, head_width)
        return res.contiguous().view(N, t, d)

# Multilayer Perceptron:
class MLP(nn.Module):
    def __init__(self, params:ModelParameters):
        super(MLP, self).__init__()
        self.p1 = nn.Linear(params.d_model, params.hidden_dim, bias=False)
        self.p2 = nn.Linear(params.hidden_dim, params.d_model, bias=False)
        # final projection before preadditive:
        self.preadditive = nn.Linear(params.d_model, params.d_model, bias=False)

    def weightTye(self, other):
        self.p1.weight = other.p1.weight
        self.p2.weight = other.p2.weight
        self.preadditive.weight = other.preadditive.weight

    def forward(self, x):
        x = self.p1(x)
        x = F.silu(x)
        x = self.p2(x)
        return x

# Main transformer decoder block:
class TBlock(nn.Module):
    def __init__(self, device, params: ModelParameters):
        super(TBlock, self).__init__()
        self.norm1 = nn.LayerNorm(params.d_model)
        self.MHSA = MHSA(device, params)
        self.MLP = MLP(params)
        self.norm2 = nn.LayerNorm(params.d_model)
        # mask stored here so that we don't have to redefine it again
        self.mask = torch.triu(torch.full((params.max_seq_len, params.max_seq_len), -float('inf')), diagonal=1)
        self.mask = self.mask.unsqueeze(0).unsqueeze(0).to(device)
    
    def weightTye(self, other):
        self.MHSA.weightTye(other.MHSA)
        self.MLP.weightTye(other.MLP)
        print("successfully tyed blocks!")

    def forward(self, x):
        # slight change: applying Norm before Attention head and MLP
        x = self.norm1(x)
        att = x + self.MHSA(x, self.mask)
        x = self.MLP(self.norm2(att)) # TODO: This is not correct maybe?
        return att+x

class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, token_count, device, params: ModelParameters):
        super(Transformer, self).__init__()
        self.params = params
        self.device = device
        self.token_count = token_count
        # embed each token into a vector of dimension d_model:
        self.embed = nn.Embedding(token_count, params.d_model)
        # Transformer blocks stored as list
        t_blocks = [TBlock(device, params) for _ in range(params.blocks)]
        # weight tying some transformer blocks (experimental):
        for el in params.weight_tying_dict:
            to_tye = t_blocks[params.weight_tying_dict[el]]
            print(f"tying block {el} with block {params.weight_tying_dict[el]}")
            t_blocks[el].weightTye(to_tye)
        self.blocks = nn.ModuleList(t_blocks)
        # final projection:
        self.last_lin = nn.Linear(params.d_model, token_count, bias=False)
        # weight tying (see for example https://benjaminwarner.dev/2023/07/28/rest-of-the-transformer):
        self.last_lin.weight = self.embed.weight
        # apply special initialization scheme to the layers
        self.apply(self._init_weights)
        

    def _init_weights(self, module):
        """
        applying xavier_normal_ as somewhat proposed by
        https://www.cs.toronto.edu/~mvolkovs/ICML2020_tfixup.pdf
        """
        if isinstance(module, nn.Linear) or isinstance(module, nn.Embedding):
            nn.init.xavier_normal_(module.weight)

    def forward(self, x, y=None):
        # shape: (N, t)
        x = self.embed(x) # embed the tokens into d_model dimensional vectors
        # Apply transformer blocks:
        # shape: (N, t, d)
        for block in self.blocks:
            x = block(x)
        x = self.last_lin(x)
        # Note that the last Softmax function is implicitly calculated in the cross entropy
        if y is not None:
            x_view = x.view(-1, self.token_count)
            y_view = y.view(-1)
            self.last_loss = F.cross_entropy(x_view, y_view)
        return x
    
    def getAttScores(self):
        return [i.MHSA.att_matr for i in self.blocks]

    def gen(self, max_tokens, inpt_tokens=None, start_t=None):
        tensors = [l for l in self.genLazy(max_tokens, inpt_tokens, start_t)]
        return tensors[-1]
    
    def genLazy(self, max_tokens, inpt_tokens=None, start_t=None, temperature=1):
        """
        generates stream of tokens based on this model
        """
        if inpt_tokens is None:
            res = torch.zeros((1, 1), dtype=torch.long).to(self.device)
            if start_t is not None:
                res[0, 0] = start_t
        else:
            res = inpt_tokens.to(self.device)
        for _ in range(max_tokens):
            if res.shape[1] >= self.params.max_seq_len:
                res = res[:, 1:]
            y = self(res)
            y = y[:,-1,:]*temperature
            probs = F.softmax(y, -1)
            next = torch.multinomial(probs, 1) # sample
            res = torch.cat((res, next), dim=1)
            yield res
        return res
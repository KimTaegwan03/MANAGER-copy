import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool, GATConv
import os
import time
import math
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModel.from_pretrained('bert-base-uncased')
model.eval()
model.to('cuda')

def get_chain_length(edge_index):
    sources = edge_index[0]
    targets = edge_index[1]
    
    length = 1  # 첫 엣지는 무조건 포함됨
    
    for i in range(1, edge_index.size(1)):
        if sources[i] == targets[i-1]:
            length += 1
        else:
            break  # 사슬이 끊기면 종료

    return length

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:,:x.size(0), :]

class CrossModalAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CrossModalAttention, self).__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=2, batch_first=True)
        self.fc = nn.Linear(input_dim, hidden_dim)

    def forward(self, x1, x2): # x1 = alpha->beta  , x2 = beta
        x1_tilde = self.norm(x1)
        x2_tilde = self.norm(x2)
        attn_output, _ = self.attention(x1_tilde, x2_tilde, x2_tilde)
        output = self.fc(attn_output)
        output = output + self.norm(output)
        return output

class MyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2):
        super(MyModel, self).__init__()
        
        self.audio_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)
        self.video_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)
        self.text_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)

        self.audio_projection = nn.Linear(hidden_dim1, hidden_dim2)
        self.video_projection = nn.Linear(hidden_dim1, hidden_dim2)
        self.text_projection = nn.Linear(hidden_dim1, hidden_dim2)
        self.question_projection = nn.Linear(hidden_dim1, hidden_dim2)

        self.pos_encoder = PositionalEncoding(hidden_dim1)
        self.text_modal_token = nn.Parameter(torch.randn(1, hidden_dim2))  # shape: [1, hidden_dim2]
        self.audio_modal_token = nn.Parameter(torch.randn(1, hidden_dim2))
        self.video_modal_token = nn.Parameter(torch.randn(1, hidden_dim2))
        self.question_modal_token = nn.Parameter(torch.randn(1, hidden_dim2))

        cls_token_id = tokenizer.cls_token_id
        cls_token = model.embeddings.word_embeddings(torch.tensor(cls_token_id).to('cuda'))  # shape: [1, hidden_dim]
        print(cls_token.shape)
        self.cls_token = nn.Parameter(cls_token.unsqueeze(0))  # shape: [1, hidden_dim]

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim1, dropout=0.2,nhead=8, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=8)

        self.fc = nn.Linear(hidden_dim1, 1)

    def forward(self, a:torch.Tensor, v, t, q): # audio, video, text, question
        # z_a = self.audio_conv(a.permute(0, 2, 1)).permute(0, 2, 1)
        # z_a = nn.ReLU()(z_a)
        # z_v = self.video_conv(v.permute(0, 2, 1)).permute(0, 2, 1)
        # z_v = nn.ReLU()(z_v)
        # z_t = self.text_conv(t.permute(0, 2, 1)).permute(0, 2, 1)
        # z_t = nn.ReLU()(z_t)
        # z_q = self.text_conv(q.permute(0, 2, 1)).permute(0, 2, 1)
        # z_q = nn.ReLU()(z_q)

        z_a = self.audio_projection(a)
        z_a = nn.ReLU()(z_a)
        z_v = self.video_projection(v)
        z_v = nn.ReLU()(z_v)
        z_t = self.text_projection(t)
        z_t = nn.ReLU()(z_t)
        z_q = self.question_projection(q)
        z_q = nn.ReLU()(z_q)

        z_a = self.pos_encoder(z_a)
        z_v = self.pos_encoder(z_v)
        z_t = self.pos_encoder(z_t)
        z_q = self.pos_encoder(z_q)

        z_a = z_a + self.audio_modal_token
        z_v = z_v + self.video_modal_token
        z_t = z_t + self.text_modal_token
        z_q = z_q + self.question_modal_token

        z = torch.cat((z_q, z_t, z_a, z_v), dim=1)  # Concatenate along the sequence dimension

        z = torch.cat((self.cls_token.expand(z.size(0), -1, -1), z), dim=1)  # Add cls token
        z = self.encoder(z)  # Apply transformer encoder
        z = z[:, 0, :]  # Take the cls token output
        z = self.fc(z)  # Apply final linear layer
        # z = torch.sigmoid(z)  # Apply sigmoid activation
        return z


    
class Monopoly(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3):
        super(Monopoly, self).__init__()

        self.audio_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)
        self.video_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)
        self.text_conv = nn.Conv1d(input_dim, hidden_dim1, kernel_size=3, padding=1)

        self.pos_encoder = PositionalEncoding(hidden_dim1)
        
        self.v_a_attention = CrossModalAttention(hidden_dim1, hidden_dim2)
        self.t_a_attention = CrossModalAttention(hidden_dim1, hidden_dim2)
        self.v_t_attention = CrossModalAttention(hidden_dim1, hidden_dim2)
        self.a_t_attention = CrossModalAttention(hidden_dim1, hidden_dim2)
        self.a_v_attention = CrossModalAttention(hidden_dim1, hidden_dim2)
        self.t_v_attention = CrossModalAttention(hidden_dim1, hidden_dim2)

        self.audio_transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim2*2, nhead=2, batch_first=True)
        self.video_transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim2*2, nhead=2, batch_first=True)
        self.text_transformer_layer = nn.TransformerEncoderLayer(d_model=hidden_dim2*2, nhead=2, batch_first=True)

        self.audio_transformer = nn.TransformerEncoder(self.audio_transformer_layer, num_layers=2)
        self.video_transformer = nn.TransformerEncoder(self.video_transformer_layer, num_layers=2)
        self.text_transformer = nn.TransformerEncoder(self.text_transformer_layer, num_layers=2)

        self.fc_a = nn.Linear(hidden_dim2*2, hidden_dim3)
        self.fc_v = nn.Linear(hidden_dim2*2, hidden_dim3)
        self.fc_t = nn.Linear(hidden_dim2*2, hidden_dim3)
        self.softmax = nn.Softmax(dim=hidden_dim3)

        self.output = nn.Linear(hidden_dim3*3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, a, v, t):
        XA = self.audio_conv(a.permute(0, 2, 1)).permute(0, 2, 1)
        XV = self.video_conv(v.permute(0, 2, 1)).permute(0, 2, 1)
        XT = self.text_conv(t.permute(0, 2, 1)).permute(0, 2, 1)

        XAPE = self.pos_encoder(XA)
        XVPE = self.pos_encoder(XV)
        XTPE = self.pos_encoder(XT) # => [B, S, hidden_dim1]

        x1 = self.v_a_attention(XAPE, XVPE)
        x2 = self.v_t_attention(XTPE, XVPE)
        x3 = self.t_a_attention(XAPE, XTPE)
        x4 = self.a_v_attention(XVPE, XAPE)
        x5 = self.a_t_attention(XTPE, XAPE)
        x6 = self.t_v_attention(XVPE, XTPE)

        ZA = torch.cat((x1,x3), dim=2)
        ZT = torch.cat((x2,x5), dim=2)
        ZV = torch.cat((x4,x6), dim=2)

        ZA = self.audio_transformer(ZA)
        ZV = self.video_transformer(ZV)
        ZT = self.text_transformer(ZT)

        WA_tilde = F.softmax(self.fc_a(ZA))
        WV_tilde = F.softmax(self.fc_v(ZV))
        WT_tilde = F.softmax(self.fc_t(ZT))

        Wall_tilde = WA_tilde + WV_tilde + WT_tilde

        WA = WA_tilde / Wall_tilde
        WV = WV_tilde / Wall_tilde
        WT = WT_tilde / Wall_tilde

        Zfused = WA*ZA + WV*ZV + WT*ZT

        out = self.output(Zfused)
        out = self.sigmoid(out)

        return out



class MANAGER(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim, num_encoder_layers, dropout_p):
        """
        Args:
            input_dim (int): Input dimension.
            hidden_dim1 (int): First hidden dimension.
            hidden_dim2 (int): Second hidden dimension.
            output_dim (int): Output dimension.
            num_encoder_layers (int): Number of encoder layers in the transformer.
            num_decoder_layers (int): Number of decoder layers in the transformer.
            dropout_p (float): Dropout probability.
        """
        super(MANAGER, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        # self.conv1 = GATConv(input_dim, hidden_dim1, heads=2, dropout=dropout_p)
        # self.conv2 = GATConv(hidden_dim1*2, hidden_dim2, heads=1, dropout=dropout_p)

        self.cls_token = nn.Parameter(torch.randn(1, hidden_dim2))  # shape: [1, hidden_dim]

        self.special_token = nn.Parameter(torch.randn(1, hidden_dim2))  # shape: [1, hidden_dim]

        self.src_mask = None
        self.pos_encoder = PositionalEncoding(hidden_dim2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim2, nhead=2, dropout=dropout_p, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        self.decoder = nn.Linear(hidden_dim2,output_dim)
        self.sig = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)


    def forward(self,src,edge,batch,title):
        utt_len = get_chain_length(edge)
        if self.src_mask is None or self.src_mask.size(0) != utt_len+2:
            device = src.device
            mask = self._generate_square_subsequent_mask(utt_len+2).to(device)
            self.src_mask = mask

        src = self.conv1(src,edge)
        src = F.relu(src)
        src = self.conv2(src,edge)
        src = F.relu(src)

        src = src[:utt_len]

        src = self.pos_encoder(src)
        src = torch.concat([title,self.special_token,src.squeeze()],dim=0)
        src = src.unsqueeze(0)
        src = self.transformer_encoder(src)[:,0,:]
        # graph_repr = global_mean_pool(output,batch=batch)
        y = self.decoder(src)
        # y = self.sig(y)
        return y
    
    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz, device = torch.device('cpu'))) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
if __name__ == "__main__":
    # Example usage
    input_dim = 768
    hidden_dim1 = 768
    hidden_dim2 = 512
    output_dim = 1
    num_encoder_layers = 2
    dropout_p = 0.1

    model = MANAGER(input_dim, hidden_dim1, hidden_dim2, output_dim, num_encoder_layers, dropout_p).to('cuda').half()
    
    # Create a dummy input tensor
    src = torch.load('data/emb_0001.pt').to(torch.float16).to('cuda')

    graph = torch.load('data/graph_0001.pt').to('cuda')
    
    # Forward pass
    output = model(src,graph.edge_index)

    print(output.shape)

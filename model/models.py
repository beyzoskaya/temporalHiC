import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from model import layers
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class STGCNChebGraphConv(nn.Module):
    # STGCNChebGraphConv contains 'TGTND TGTND TNFF' structure
    # ChebGraphConv is the graph convolution from ChebyNet.
    # Using the Chebyshev polynomials of the first kind as a graph filter.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (ChebGraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, blocks, n_vertex):
        super(STGCNChebGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
            x = x * 1.5
        
        return x
    
class ScaledSTGCNChebGraphConv(STGCNChebGraphConv):
    def __init__(self, args, blocks, n_vertex):
        super(ScaledSTGCNChebGraphConv, self).__init__(args, blocks, n_vertex)
        # Add more aggressive scaling parameters
        self.scale_factor = nn.Parameter(torch.ones(1) * 3.0)  # Start with larger scale
        self.shift = nn.Parameter(torch.zeros(1))  # Allow for shift
        
    def forward(self, x):
        # Get base predictions
        x = super().forward(x)
        
        # Get the previous time step values
        prev_values = x[:, :, -2:-1, :] if x.size(2) > 1 else 0
        
        # Calculate changes from previous step
        changes = x[:, :, -1:, :] - prev_values
        
        # Scale the changes more aggressively
        scaled_changes = changes * self.scale_factor + self.shift
        
        # Apply scaled changes to previous values
        x = prev_values + scaled_changes
        
        return x

class STGCNChebGraphConvProjected(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(STGCNChebGraphConvProjected, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlockLSTM(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            #self.relu = nn.ReLU()
            self.elu = nn.ELU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        #FIXED projection layer is added for prediction of expression values
        self.expression_proj = nn.Sequential(
        nn.Linear(blocks[-1][0], 32), 
        nn.ELU(),
        nn.Dropout(p=0.1),
        nn.Linear(32, 16),
        nn.ELU(),
        nn.Linear(16, 1)
        )

        self.expression_proj_miRNA = nn.Sequential(
            nn.Linear(blocks[-1][0], 64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64,32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32,16),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(16,1)
        )
        
    def forward(self, x):
        #print("Input Shape:", x.shape)
        # Original STGCN forward pass
        x = self.st_blocks(x)
        #print("After STConvBlockLSTM Shape:", x.shape)
        
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.elu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        #print("After OutputBlock Shape:", x.shape)

        # Project to expression values
        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj_miRNA(x)  # [batch, time_steps, nodes, 1]
        #print("After Projection Shape:", x.shape)
        x = x.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        return x 
    
class STGCNChebGraphConvProjectedGeneConnectedAttention(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STGCNChebGraphConvProjectedGeneConnectedAttention, self).__init__()
        
        connections = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)], 
                                 dtype=torch.float32)  # not implemented for Long error
        self.connection_weights = F.softmax(connections, dim=0)
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko

        self.connectivity_attention = nn.Sequential(
            nn.Linear(blocks[-1][0], blocks[-1][0]//2),
            nn.LayerNorm(blocks[-1][0]//2 ),
            nn.ELU(),
            nn.Linear(blocks[-1][0]//2, 1),
            nn.Sigmoid()
        )
        
        self.attention_scale = nn.Parameter(torch.tensor(0.1))

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            #self.elu = nn.ELU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        identity = x
        
        x = self.st_blocks(x)
        
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.elu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        
        learned_attention = self.connectivity_attention(x)  # [batch, time_steps, nodes, 1]
      
        connectivity_weights = self.connection_weights
        connectivity_weights = connectivity_weights.view(1, 1, -1, 1)
        
        attention = (learned_attention * (1.0 + self.attention_scale * connectivity_weights))
        attention = F.layer_norm(attention, [attention.size(-1)])

        x = x * attention + 0.1 * x  # Small residual to maintain variation
        
        x = self.expression_proj(x)
        x = x.permute(0, 3, 1, 2)
        
        return x

# This model works best for mRNA predictions with enhanced_temporal_loss
class STGCNChebGraphConvProjectedGeneConnectedAttentionLSTM(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STGCNChebGraphConvProjectedGeneConnectedAttentionLSTM, self).__init__()
        
        connections = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)], 
                                 dtype=torch.float32)
        self.connection_weights = F.softmax(connections, dim=0)
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko

        # bidirectional LSTM 
        self.lstm = nn.LSTM(
            input_size=blocks[-3][-1],  # Input size is the feature dimension
            hidden_size=blocks[-3][-1],
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        self.lstm_mirna = nn.LSTM(
            input_size=blocks[-3][-1],  # Now it is 64 for miRNA
            hidden_size=blocks[-3][-1],
            num_layers=4,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )

        print(f"Hidden size blocks [-3][-1]: {blocks[-3][-1]}")
    
        self.lstm_proj = nn.Linear(2 * blocks[-3][-1], blocks[-3][-1])  # *2 for bidirectional
        
        self.lstm_norm = nn.LayerNorm([n_vertex, blocks[-3][-1]])
        
        self.connectivity_attention = nn.Sequential(
            nn.Linear(blocks[-1][0], blocks[-1][0]//2),
            nn.LayerNorm(blocks[-1][0]//2),
            nn.ELU(),
            nn.Linear(blocks[-1][0]//2, 1),
            nn.Sigmoid()
        )

        self.connectivity_attention_mirna = nn.Sequential(
            nn.Linear(blocks[-1][0], blocks[-1][0]),
            nn.LayerNorm(blocks[-1][0]),
            nn.ELU(),
            nn.Linear(blocks[-1][0], 1),
            nn.Sigmoid()
        )
        
        self.attention_scale = nn.Parameter(torch.tensor(0.1))

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 1)
        )

        self.expression_proj_mirna = nn.Sequential(
            nn.Linear(blocks[-1][0], 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        identity = x
        
        # ST-Blocks processing
        x = self.st_blocks(x)
        
        batch_size, features, time_steps, nodes = x.shape
        x_lstm = x.permute(0, 3, 2, 1)  # [batch, nodes, time_steps, features]
        x_lstm = x_lstm.reshape(batch_size * nodes, time_steps, features)
        
        lstm_out, _ = self.lstm_mirna(x_lstm)
        lstm_out = self.lstm_proj(lstm_out)
        lstm_out = lstm_out.reshape(batch_size, nodes, time_steps, features)
        lstm_out = lstm_out.permute(0, 3, 2, 1)  # [batch, features, time_steps, nodes]
        
        # Residual connection with ST-Blocks output
        x = x + lstm_out
        x = self.lstm_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        # Attention mechanism
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        learned_attention = self.connectivity_attention(x)
        
        connectivity_weights = self.connection_weights.view(1, 1, -1, 1)
        attention = (learned_attention * (1.0 + self.attention_scale * connectivity_weights))
        attention = F.layer_norm(attention, [attention.size(-1)])
        
        x = x * attention + 0.1 * x  # Residual connection
        
        # Final projection
        x = self.expression_proj_mirna(x)
        x = x.permute(0, 3, 1, 2)
        
        return x

class STGCNChebGraphConvProjectedGeneConnectedMultiHeadAttentionLSTMmirna(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STGCNChebGraphConvProjectedGeneConnectedMultiHeadAttentionLSTMmirna, self).__init__()
        
        connections = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)], 
                                 dtype=torch.float32)
        self.connection_weights = F.softmax(connections, dim=0)

        modules = []
        for l in range(len(blocks) - 3):
            # I am trying without attention STConvBlock layer because tempconv and graphconv already have attention
            modules.append(layers.STConvBlockTwoSTBlocks(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)

        self.global_attention_scale = nn.Parameter(torch.tensor(0.4)) # I add this for the residual amount as a learnable parameter because I got worse results with adding residual directly

        """
        Ko represents the remaining length of the temporal sequence after it passes through all the spatial-temporal convolutional blocks
        """  
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        print(f"Ko: {self.Ko}")

        # I changed dropour from 0.2 to 0.3 lastly
        self.lstm_mirna = nn.LSTM(
            input_size=blocks[-3][-1], 
            hidden_size=blocks[-3][-1],
            #num_layers=6,
            num_layers=4,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
            #dropout=0.3
        )

        print(f"Hidden size blocks [-3][-1]: {blocks[-3][-1]}")
    
        self.lstm_proj = nn.Linear(2 * blocks[-3][-1], blocks[-3][-1])  # *2 for bidirectional
        
        self.lstm_norm = nn.LayerNorm([n_vertex, blocks[-3][-1]])

        print(f"Dimension fo embed_dim in multihead attention: {blocks[-1][0]}")

        # I changed dropout 0.2 to 0.3 lastly 
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=blocks[-1][0],  # Feature dimension after output block
            #num_heads=6,
            num_heads=4,
            #dropout=0.1
            dropout=0.3
        )

        self.attention_scale = nn.Parameter(torch.tensor(0.1))

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        
        #if self.Ko > 1:
        #    self.output = layers.TemporalEnhancedOutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
        #                           n_vertex, args.act_func, args.enable_bias, args.droprate)

        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

        self.expression_proj_mirna = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
    
        self._init_weights()

        
    def forward(self, x):
        #print(f"Shape of x: {x.shape}") # [batch=1, embedding_dim=128, sequence_len=5, nodes=162]
        x = self.st_blocks(x)
        #print(f"Shape after ST blocks: {x.shape}") 

        batch_size, features, time_steps, nodes = x.shape
        #print(f"Batch size: {batch_size}, Features: {features}, Time steps: {time_steps}, Nodes: {nodes}") 

        # LSTM processing
        x_lstm = x.permute(0, 3, 2, 1)  # [batch, nodes, time_steps, features]
        #print(f"Shape after permute for LSTM: {x_lstm.shape}")

        x_lstm = x_lstm.reshape(batch_size * nodes, time_steps, features)
        #print(f"Shape before LSTM: {x_lstm.shape}")
        
        lstm_out, _ = self.lstm_mirna(x_lstm)
        #print(f"Shape after LSTM: {lstm_out.shape}")

        lstm_out = self.lstm_proj(lstm_out)
        lstm_out = lstm_out.reshape(batch_size, nodes, time_steps, features)
        #print(f"Shape after LSTM projection: {lstm_out.shape}")
        lstm_out = lstm_out.permute(0, 3, 2, 1)  # [batch, features, time_steps, nodes]
        #print(f"Shape after permute back: {lstm_out.shape}")
        
        # Residual connection  
        x = x + lstm_out
        x = self.lstm_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        # Output block processing
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        #print(f"Shape before attention: {x.shape}")
        
        # Get current dimensions after output processing
        _, current_features, time_steps, nodes = x.shape
        
        x_attention = x.permute(2, 0, 3, 1)  # [time_steps, batch, nodes, features]
        x_attention = x_attention.reshape(time_steps, batch_size * nodes, current_features)
        #print(f"x_attention shape: {x_attention.shape}") --> [5, 162, 96]
        
        attn_output, _ = self.multihead_attention(x_attention, x_attention, x_attention)

        attn_output = attn_output.reshape(time_steps, batch_size, nodes, current_features)
        attn_output = attn_output.permute(1, 3, 0, 2)  # [batch, features, time_steps, nodes]
    
        x = x + self.global_attention_scale * attn_output 
        #x = x + 0.2 * attn_output
        
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj_mirna(x)
        #print(f"Shape after expression projection: {x.shape}")
        x = x.permute(0, 3, 1, 2)  # [batch, features, time_steps, nodes]
        
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

class STGCNChebGraphConvProjectedGeneConnectedMultiHeadAttentionLSTMmirnaWithNumberOfConnections(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STGCNChebGraphConvProjectedGeneConnectedMultiHeadAttentionLSTMmirnaWithNumberOfConnections, self).__init__()
    
        self.connections_raw = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)], 
                                 dtype=torch.float32)
        self.connection_weights = F.softmax(self.connections_raw, dim=0)
        
        self.genes_with_27_connections = torch.where(self.connections_raw == 27.0)[0]
        print(f"Genes with exactly 27 connections: {self.genes_with_27_connections.tolist()}")
        
        self.lstm_standard = nn.LSTM(
            input_size=blocks[-3][-1], 
            hidden_size=blocks[-3][-1],
            num_layers=4,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.lstm_special = nn.LSTM(
            input_size=blocks[-3][-1], 
            hidden_size=blocks[-3][-1],
            num_layers=6,  # More layers for genes with 27 connections
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.attention_standard = nn.MultiheadAttention(
            embed_dim=blocks[-1][0],
            num_heads=4,
            dropout=0.3
        )
        
        self.attention_special = nn.MultiheadAttention(
            embed_dim=blocks[-1][0],
            num_heads=6,  # More heads for genes with 27 connections
            dropout=0.3
        )

        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlockTwoSTBlocks(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)

        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        print(f"Ko: {self.Ko}")

        print(f"Hidden size blocks [-3][-1]: {blocks[-3][-1]}")
    
        self.lstm_proj = nn.Linear(2 * blocks[-3][-1], blocks[-3][-1])  # *2 for bidirectional
        
        self.lstm_norm = nn.LayerNorm([n_vertex, blocks[-3][-1]])

        print(f"Dimension fo embed_dim in multihead attention: {blocks[-1][0]}")

        self.attention_scale = nn.Parameter(torch.tensor(0.1))

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

        self.expression_proj_mirna = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
    
        self._init_weights()

    def forward(self, x):
        x = self.st_blocks(x)
        
        batch_size, features, time_steps, nodes = x.shape
        
        x_lstm = x.permute(0, 3, 2, 1)  # [batch, nodes, time_steps, features]
        x_lstm = x_lstm.reshape(batch_size * nodes, time_steps, features)
        #print(f"x_lstm shape: {x_lstm.shape}")

        lstm_out = torch.zeros(batch_size * nodes, time_steps, features * 2)

        for i in range(nodes):
            indices = torch.arange(i, batch_size * nodes, nodes)
            
            gene_data = x_lstm[indices]
            #print(f"gene_data shape: {gene_data.shape}")
            
            if i in self.genes_with_27_connections:
                gene_lstm_out, _ = self.lstm_special(gene_data)
                #print(f"gene_lstm_out shape for 27 connected genes: {gene_lstm_out.shape}")
            else:
                gene_lstm_out, _ = self.lstm_standard(gene_data)
                #print(f"gene_lstm_out shape for other genes: {gene_lstm_out.shape}")
            
            lstm_out[indices] = gene_lstm_out
            #print(f"lstm_out shape before projection: {lstm_out.shape}")
  
        lstm_out = self.lstm_proj(lstm_out)  # This reduces from 2*features to features
        #print(f"lstm_out shape after projection: {lstm_out.shape}")
        
        lstm_out = lstm_out.reshape(batch_size, nodes, time_steps, features)
        lstm_out = lstm_out.permute(0, 3, 2, 1)
        
        x = x + lstm_out
        x = self.lstm_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
 
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
 
        _, current_features, time_steps, nodes = x.shape
        
        attn_outputs = []
        for t in range(time_steps):
            time_attn_outputs = []
            for node_idx in range(nodes):
              
                node_features = x[:, :, t, node_idx].unsqueeze(0)  # [1, batch, features]
                
                if node_idx in self.genes_with_27_connections:
                    attn_out, _ = self.attention_special(node_features, node_features, node_features)
                else:
                    attn_out, _ = self.attention_standard(node_features, node_features, node_features)
                    
                time_attn_outputs.append(attn_out)
            
            time_attn_out = torch.cat(time_attn_outputs, dim=0)  # [nodes, batch, features]
            attn_outputs.append(time_attn_out)

        attn_output = torch.stack(attn_outputs, dim=0)  # [time_steps, nodes, batch, features]
        attn_output = attn_output.permute(2, 3, 0, 1)  # [batch, features, time_steps, nodes]
      
        x = x + 0.4 * attn_output
        
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj_mirna(x)
        x = x.permute(0, 3, 1, 2)  # [batch, features, time_steps, nodes]
        
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0)

"""
model call parameters while training
model = BiLSTMExpressionPredictor(
    embedding_dim=256,  
    hidden_dim=128,     
    n_vertex=dataset.num_nodes,
    num_layers=6,    
    dropout=0.3        
    )
"""
class BiLSTMExpressionPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_vertex, num_layers=6, dropout=0.3):
        super(BiLSTMExpressionPredictor, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_vertex = n_vertex

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

        self.projection = nn.Linear(2 * hidden_dim, hidden_dim)

        self.expression_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # input shape: [batch, features, time_steps, nodes]
        batch_size, features, time_steps, nodes = x.shape
        
        x = x.permute(0, 3, 2, 1)  # [batch, nodes, time_steps, features]
        x = x.reshape(batch_size * nodes, time_steps, features)
        
        lstm_out, _ = self.lstm(x)
        lstm_out = self.projection(lstm_out)
        
        lstm_out = lstm_out.reshape(batch_size, nodes, time_steps, self.hidden_dim)
        
        # From [batch, nodes, time_steps, hidden_dim] to [batch, time_steps, nodes, hidden_dim]
        x = lstm_out.permute(0, 2, 1, 3)  # [batch, time_steps, nodes, hidden_dim]
      
        x = self.expression_predictor(x)  # [batch, time_steps, nodes, 1]
    
        x = x.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        return x

"""
model parameters when calling in training
model = MultiHeadAttentionPredictor(
    embedding_dim=256, 
    hidden_dim=128,     
    n_vertex=dataset.num_nodes,
    num_heads=6,        
    dropout=0.2        
)
"""
class MultiHeadAttentionPredictor(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, n_vertex, num_heads=6, dropout=0.3):
        super(MultiHeadAttentionPredictor, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_vertex = n_vertex
       
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)

        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
    
        self.projection = nn.Linear(hidden_dim, hidden_dim)
        
        self.expression_predictor = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Input shape: [batch=1, features, time_steps, nodes]
        batch_size, features, time_steps, nodes = x.shape
        
        x = x.permute(0, 3, 2, 1)  # [1, nodes, time_steps, features]
        x_flat = x.reshape(batch_size * nodes, time_steps, features)
        x_projected = self.input_projection(x_flat)  # [nodes, time_steps, hidden_dim]
        
        x_for_attn = x_projected.permute(1, 0, 2)  # [time_steps, nodes, hidden_dim]

        attn_output, _ = self.multihead_attention(x_for_attn, x_for_attn, x_for_attn)
        
        attn_output = attn_output.permute(1, 0, 2)  # [nodes, time_steps, hidden_dim]
        
        projected_output = self.projection(attn_output)
  
        output = projected_output.reshape(batch_size, nodes, time_steps, self.hidden_dim)
        
        # From [batch, nodes, time_steps, hidden_dim] to [batch, time_steps, nodes, hidden_dim]
        output = output.permute(0, 2, 1, 3)  # [1, time_steps, nodes, hidden_dim]
       
        predictions = self.expression_predictor(output)  # [1, time_steps, nodes, 1]
        
        predictions = predictions.permute(0, 3, 1, 2)  # [1, 1, time_steps, nodes]
        
        return predictions

class STGCNChebGraphConvProjectedGeneConnectedTransformerAttentionMirna(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STGCNChebGraphConvProjectedGeneConnectedTransformerAttentionMirna, self).__init__()
        
        connections = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)], 
                                 dtype=torch.float32)
        self.connection_weights = F.softmax(connections, dim=0)
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlockTwoSTBlocks(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=blocks[-3][-1],
            nhead=4, 
            dim_feedforward=4 * blocks[-3][-1], 
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,  
        )

        print(f"Hidden size blocks [-3][-1]: {blocks[-3][-1]}")
        
        self.transformer_norm = nn.LayerNorm([n_vertex, blocks[-3][-1]])
 
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=blocks[-1][0],  # Feature dimension after output block
            num_heads=4,
            dropout=0.1
        )

        self.attention_scale = nn.Parameter(torch.tensor(0.1))

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

        self.expression_proj_mirna = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
    
        self._init_weights()

        
    def forward(self, x):
        x = self.st_blocks(x)
        
        batch_size, features, time_steps, nodes = x.shape
        
        x_trans = x.permute(0, 3, 2, 1)  # [batch, nodes, time_steps, features]
        x_trans = x_trans.reshape(batch_size * nodes, time_steps, features)
        
        trans_out = self.transformer_encoder(x_trans)
        
        trans_out = trans_out.reshape(batch_size, nodes, time_steps, features)
        trans_out = trans_out.permute(0, 3, 2, 1)  # [batch, features, time_steps, nodes]
        
        x = x + trans_out
        x = self.transformer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        _, current_features, time_steps, nodes = x.shape
        
        x_attention = x.permute(2, 0, 3, 1)  # [time_steps, batch, nodes, features]
        x_attention = x_attention.reshape(time_steps, batch_size * nodes, current_features)
        
        attn_output, _ = self.multihead_attention(x_attention, x_attention, x_attention)

        attn_output = attn_output.reshape(time_steps, batch_size, nodes, current_features)
        attn_output = attn_output.permute(1, 3, 0, 2)  # [batch, features, time_steps, nodes]
        
        x = x + 0.5 * attn_output  
        
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj_mirna(x)
        x = x.permute(0, 3, 1, 2)  # [batch, features, time_steps, nodes]
        
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.TransformerEncoder) or isinstance(m, nn.TransformerEncoderLayer):
                for name, param in m.named_parameters():
                    if 'weight' in name and len(param.shape) >= 2:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name or len(param.shape) < 2:
                        nn.init.constant_(param, 0)

class STGCNChebGraphConvProjectedGeneConnectedTransformerAttentionMirnaConnectionWeights(nn.Module):
    def __init__(self, args, blocks, n_vertex, gene_connections):
        super(STGCNChebGraphConvProjectedGeneConnectedTransformerAttentionMirnaConnectionWeights, self).__init__()
        
        connections = torch.tensor([gene_connections.get(i, 0) for i in range(n_vertex)], 
                                 dtype=torch.float32)
        self.connection_weights = F.softmax(connections, dim=0)
        
        self.connection_embedding = nn.Linear(1, blocks[-3][-1] // 2)
        
        self.n_vertex = n_vertex
        
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlockTwoSTBlocks(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                            args.act_func, args.graph_conv_type, args.gso, 
                                            args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko

        self.pre_transformer = nn.Sequential(
            nn.Linear(blocks[-3][-1] + blocks[-3][-1] // 2, blocks[-3][-1]),
            nn.LayerNorm(blocks[-3][-1]),
            nn.ELU()
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=blocks[-3][-1],
            nhead=4, 
            dim_feedforward=4 * blocks[-3][-1], 
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=4,  
        )

        print(f"Hidden size blocks [-3][-1]: {blocks[-3][-1]}")
        
        self.transformer_norm = nn.LayerNorm([n_vertex, blocks[-3][-1]])
 
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=blocks[-1][0],  
            num_heads=4,
            dropout=0.1
        )

        self.attention_scale = nn.Parameter(torch.tensor(0.2))  

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)

        self.expression_proj_mirna = nn.Sequential(
            nn.Linear(blocks[-1][0], 32),
            nn.LayerNorm(32),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(32, 16),
            nn.LayerNorm(16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
    
        self.conn_attention = nn.Sequential(
            nn.Linear(blocks[-1][0], blocks[-1][0]),
            nn.LayerNorm(blocks[-1][0]),
            nn.ELU(),
            nn.Linear(blocks[-1][0], 1),
            nn.Sigmoid()
        )
    
        self._init_weights()

        
    def forward(self, x):
        x = self.st_blocks(x)
        
        batch_size, features, time_steps, nodes = x.shape
        #print(f"x shape: {x.shape}")
        #print(f"Batch size: {batch_size}, Features: {features}, Time steps: {time_steps}, Nodes: {nodes}")
        
        conn_weights = self.connection_weights.unsqueeze(1)  # [nodes, 1]
        conn_embedding = self.connection_embedding(conn_weights)  # [nodes, features//2]
     
        x_trans = x.permute(0, 3, 2, 1)  # [batch, nodes, time_steps, features]
     
        conn_embedding = conn_embedding.unsqueeze(0).unsqueeze(2)  # [1, nodes, 1, features//2]
        conn_embedding = conn_embedding.expand(batch_size, -1, time_steps, -1)  # [batch, nodes, time_steps, features//2]
    
        x_combined = torch.cat([x_trans, conn_embedding], dim=-1)  # [batch, nodes, time_steps, features+features//2]
        
        x_combined = self.pre_transformer(x_combined)  # [batch, nodes, time_steps, features]
        
        x_combined = x_combined.reshape(batch_size * nodes, time_steps, features)
        
        trans_out = self.transformer_encoder(x_combined)
        
        trans_out = trans_out.reshape(batch_size, nodes, time_steps, features)
        trans_out = trans_out.permute(0, 3, 2, 1)  # [batch, features, time_steps, nodes]
        
        x = x + trans_out
        x = self.transformer_norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        _, current_features, time_steps, nodes = x.shape
        
        # multihead attention
        x_attention = x.permute(2, 0, 3, 1)  # [time_steps, batch, nodes, features]
        x_attention = x_attention.reshape(time_steps, batch_size * nodes, current_features)
        
        attn_output, _ = self.multihead_attention(x_attention, x_attention, x_attention)

        attn_output = attn_output.reshape(time_steps, batch_size, nodes, current_features)
        attn_output = attn_output.permute(1, 3, 0, 2)  # [batch, features, time_steps, nodes]

        x_conn = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        conn_attn = self.conn_attention(x_conn)  # [batch, time_steps, nodes, 1]
        
        # connection weights to attention
        conn_weights = self.connection_weights.view(1, 1, -1, 1)  # [1, 1, nodes, 1]
        modulated_conn_attn = conn_attn * (1.0 + self.attention_scale * conn_weights)
        
        modulated_conn_attn = modulated_conn_attn.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        x = x * modulated_conn_attn + 0.4 * attn_output
        
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj_mirna(x)
        x = x.permute(0, 3, 1, 2)  # [batch, features, time_steps, nodes]
        
        return x
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.TransformerEncoder) or isinstance(m, nn.TransformerEncoderLayer):
                for name, param in m.named_parameters():
                    if 'weight' in name and len(param.shape) >= 2:
                        nn.init.xavier_normal_(param)
                    elif 'bias' in name or len(param.shape) < 2:
                        nn.init.constant_(param, 0)
    
class STGCNGraphConv(nn.Module):
    # STGCNGraphConv contains 'TGTND TGTND TNFF' structure
    # GraphConv is the graph convolution from GCN.
    # GraphConv is not the first-order ChebConv, because the renormalization trick is adopted.
    # Be careful about over-smoothing.
        
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # G: Graph Convolution Layer (GraphConv)
    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normolization
    # D: Dropout

    # T: Gated Temporal Convolution Layer (GLU or GTU)
    # N: Layer Normalization
    # F: Fully-Connected Layer
    # F: Fully-Connected Layer

    def __init__(self, args, blocks, n_vertex):
        super(STGCNGraphConv, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.do = nn.Dropout(p=args.droprate)

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        return x


class STGCNGraphConvProjected(nn.Module):
   
    def __init__(self, args, blocks, n_vertex):
        super(STGCNGraphConvProjected, self).__init__()
        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STConvBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], args.act_func, args.graph_conv_type, args.gso, args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.do = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
        nn.Linear(blocks[-1][0], 16),  # Wider first projection
        nn.ReLU(),
        nn.Linear(16, 8),            # Gradual reduction
        nn.ReLU(),
        nn.Linear(8, 1)              # Final projection
    )
        

    def forward(self, x):
        x = self.st_blocks(x)
        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj(x)  # [batch, time_steps, nodes, 1]
        x = x.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        return x

class EnhancedSTGCNChebGraphConvProjected(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(EnhancedSTGCNChebGraphConvProjected, self).__init__()

        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.STAttentionBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                          args.act_func, args.graph_conv_type, args.gso, 
                                          args.enable_bias, args.droprate))
        
        self.st_blocks = nn.Sequential(*modules)

        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        print(f"Ko: {Ko}")
        self.Ko = Ko

        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        elif self.Ko == 0:
            self.fc1 = nn.Linear(in_features=blocks[-3][-1], out_features=blocks[-2][0], 
                                bias=args.enable_bias)
            self.fc2 = nn.Linear(in_features=blocks[-2][0], out_features=blocks[-1][0], 
                                bias=args.enable_bias)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
        nn.Linear(blocks[-1][0], 16),  # Wider first projection
        nn.ReLU(),
        nn.Linear(16, 8),            # Gradual reduction
        nn.ReLU(),
        nn.Linear(8, 1)              # Final projection
    )
    
    def forward(self,x):

        x = self.st_blocks(x)

        if self.Ko > 1:
            #print(f"I am in Ko>1 part")
            x = self.output(x)
        elif self.Ko == 0:
            #print(f"I am in Ko==0 part")
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = self.relu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)

        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)  # [batch, time_steps, nodes, features]
        x = self.expression_proj(x)  # [batch, time_steps, nodes, 1]
        x = x.permute(0, 3, 1, 2)  # [batch, 1, time_steps, nodes]
        
        return x


class SmallSTGCN(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(SmallSTGCN, self).__init__()

        modules = []
        for l in range(len(blocks) - 3):
            modules.append(layers.SmallSTBlock(args.Kt, args.Ks, n_vertex, blocks[l][-1], blocks[l+1], 
                                          args.act_func, args.graph_conv_type, args.gso, 
                                          args.enable_bias, args.droprate))
        self.st_blocks = nn.Sequential(*modules)

        Ko = args.n_his - (len(blocks) - 3) * 2 * (min(args.Kt, 2) - 1)
        self.Ko = Ko
        
        if self.Ko > 1:
            self.output = layers.OutputBlock(Ko, blocks[-3][-1], blocks[-2], blocks[-1][0], 
                                           n_vertex, args.act_func, args.enable_bias, args.droprate)
        else:
            self.fc1 = nn.Linear(blocks[-3][-1], blocks[-2][0], bias=args.enable_bias)
            self.fc2 = nn.Linear(blocks[-2][0], blocks[-1][0], bias=args.enable_bias)
            self.dropout = nn.Dropout(p=args.droprate)
        
        self.expression_proj = nn.Sequential(
        nn.Linear(blocks[-1][0], 8),  # Wider first projection
        nn.ReLU(),
        #nn.Linear(16, 8),            # Gradual reduction
        #nn.ReLU(),
        nn.Linear(8, 1)              # Final projection
    )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.st_blocks(x)
        
        if self.Ko > 1:
            x = self.output(x)
        else:
            x = self.fc1(x.permute(0, 2, 3, 1))
            x = F.elu(x)
            x = self.dropout(x)
            x = self.fc2(x).permute(0, 3, 1, 2)
        
        batch_size, features, time_steps, nodes = x.shape
        x = x.permute(0, 2, 3, 1)
        x = self.expression_proj(x)
        x = x.permute(0, 3, 1, 2)
        
        return x
    
class STGCNChebGraphConvWithAttentionMiRNA(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        super(STGCNChebGraphConvWithAttentionMiRNA, self).__init__()
        
        self.base_model = STGCNChebGraphConvProjected(args, blocks, n_vertex)
        
        self.temporal_attention = nn.Sequential(
            nn.Linear(blocks[-1][0], blocks[-1][0]),
            nn.Tanh(),
            nn.Linear(blocks[-1][0], 1)
        )
        
    def forward(self, x):
        base_output = self.base_model(x)
    
        batch_size, channels, time_steps, nodes = base_output.shape
        
        features = self.base_model.st_blocks(x)
        if self.base_model.Ko > 1:
            features = self.base_model.output(features)
        elif self.base_model.Ko == 0:
            features = self.base_model.fc1(features.permute(0, 2, 3, 1))
            features = self.base_model.elu(features)
            features = self.base_model.fc2(features).permute(0, 3, 1, 2)
        
        features = features.permute(0, 2, 3, 1)
        
        attention_weights = self.temporal_attention(features)  # [batch, time_steps, nodes, 1]
        attention_weights = F.softmax(attention_weights, dim=1)  # Normalize along time dimension
        
        weighted_output = base_output * attention_weights.permute(0, 3, 1, 2)
        
        return weighted_output




import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import *
from model.models import * 


class BaselineGCN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim, num_layers=2):
        super(BaselineGCN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(embedding_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden_dim, hidden_dim))
        self.layers.append(GCNConv(hidden_dim, out_dim))

    def forward(self, x, edge_index):
        # x: [num_nodes, embedding_dim]
        for conv in self.layers[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = self.layers[-1](x, edge_index)  # final layer, no activation
        return x  # shape: [num_nodes, out_dim]


class STGCN(nn.Module):
    def __init__(self, args, blocks, n_vertex):
        """
        args: args with attributes like Kt, Ks, act_func, graph_conv_type, gso, enable_bias, droprate, n_his
        blocks: list of lists, e.g., [[32,32,32], [32,48,48], [48,32,32], [32], [1]]
        n_vertex: number of graph nodes
        """
        super(STGCN, self).__init__()

        modules = []
        for l in range(len(blocks) - 3):
            modules.append(
                layers.STConvBlockTwoSTBlocks(
                    Kt=args.Kt,
                    Ks=args.Ks,
                    n_vertex=n_vertex,
                    last_block_channel=blocks[l][-1],
                    channels=blocks[l+1],
                    act_func=args.act_func,
                    graph_conv_type=args.graph_conv_type,
                    gso=args.gso,
                    bias=args.enable_bias,
                    droprate=args.droprate
                )
            )
        self.st_blocks = nn.Sequential(*modules)
        print(f"Len of blocks: {len(blocks)}")
        Ko = args.n_his - (len(blocks) - 3) * 2 * (args.Kt - 1)
        self.Ko = Ko
        print(f"[DEBUG] Ko: {self.Ko}")

        if self.Ko > 1:
            # last_block_channel matching final st_block output channels:
            last_block_channel = blocks[-2][-1]  # e.g., 48
            self.output = layers.OutputBlock(
                self.Ko,
                last_block_channel,
                blocks[-2],        # e.g., [48,32,32]
                blocks[-1][0], # final output channels
                n_vertex,
                args.act_func,
                args.enable_bias,
                args.droprate
            )

        elif self.Ko == 0:
            self.fc1 = nn.Linear(
                in_features=blocks[-2][-1],
                out_features=blocks[-2][0],
                bias=args.enable_bias
            )
            self.fc2 = nn.Linear(
                in_features=blocks[-2][0],
                out_features=blocks[-1][0],
                bias=args.enable_bias
            )
            self.elu = nn.ELU()

        self.expression_fc = nn.Linear(blocks[-1][0], 1)

    def forward(self, x):
        # x shape: [batch, channels, time, nodes]
        #print(f"[DEBUG] Input x shape: {x.shape}") # --> ([1, 32, 3, 50]) [batch, channels, time, nodes]

        # Apply spatial-temporal conv blocks
        x = self.st_blocks(x)
        #print(f"[DEBUG] After ST blocks: {x.shape}") # --> ([1, 48, 3, 50]) [batch, channels, time, nodes]

        if self.Ko > 1:
            x = self.output(x)
        elif self.Ko == 0:
            # flatten and fc
            x = self.fc1(x.permute(0, 2, 3, 1))  # [B, T, N, C]
            x = self.elu(x)
            x = self.fc2(x).permute(0, 3, 1, 2)  # back to [B, C, T, N]

        # [batch, channels, time, nodes] → [batch, time, nodes, channels]
        x = x.permute(0, 2, 3, 1)

        # final prediction: reduce feature dim to 1
        x = self.expression_fc(x)  # [B, T, N, 1]

        # [B, T, N, 1] → [B, 1, T, N]
        x = x.permute(0, 3, 1, 2)

        return x
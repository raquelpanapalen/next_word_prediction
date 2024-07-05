from xlstm import (
    xLSTMLMModel,
    xLSTMLMModelConfig,
    mLSTMBlockConfig,
    sLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
import torch.nn as nn


class xLSTM(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, seq_len, out_features, device, dropout=0.1
    ):
        super(xLSTM, self).__init__()
        mlstm_config = mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
            )
        )
        slstm_config = sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend=device,
                num_heads=4,
                conv1d_kernel_size=4,
                bias_init="powerlaw_blockdependent",
            ),
            feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
        )
        self.config = xLSTMLMModelConfig(
            mlstm_block=mlstm_config,
            slstm_block=slstm_config,
            context_length=seq_len,
            vocab_size=vocab_size,
            embedding_dim=embed_dim,
            out_features=out_features,
            dropout=dropout,
        )
        self.model = xLSTMLMModel(config=self.config)

    def forward(self, x):
        return self.model(x)

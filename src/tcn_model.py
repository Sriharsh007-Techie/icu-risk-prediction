import torch

import torch.nn as nn



# ==================================================
# Attention Block
# ==================================================

class AttentionBlock(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attention = nn.Linear(
            hidden_dim,
            1
        )


    def forward(self, x):

        attention_scores = self.attention(x)

        attention_weights = torch.softmax(
            attention_scores,
            dim=1
        )

        attended = x * attention_weights

        attended = attended.sum(dim=1)

        return attended



# ==================================================
# Residual Temporal Block
# ==================================================

class TemporalBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout=0.2
    ):

        super().__init__()


        padding = (
            (kernel_size - 1)
            * dilation
        ) // 2


        # ------------------------------------------
        # Conv Layer 1
        # ------------------------------------------

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn1 = nn.BatchNorm1d(
            out_channels
        )



        # ------------------------------------------
        # Conv Layer 2
        # ------------------------------------------

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation
        )

        self.bn2 = nn.BatchNorm1d(
            out_channels
        )



        # ------------------------------------------
        # Activation + Dropout
        # ------------------------------------------

        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(
            dropout
        )



        # ------------------------------------------
        # Residual Connection
        # ------------------------------------------

        self.downsample = None

        if in_channels != out_channels:

            self.downsample = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1
            )



    def forward(self, x):

        residual = x


        # ------------------------------------------
        # First Conv Block
        # ------------------------------------------

        out = self.conv1(x)

        out = self.bn1(out)

        out = self.relu(out)

        out = self.dropout(out)



        # ------------------------------------------
        # Second Conv Block
        # ------------------------------------------

        out = self.conv2(out)

        out = self.bn2(out)

        out = self.relu(out)

        out = self.dropout(out)



        # ------------------------------------------
        # Residual Path
        # ------------------------------------------

        if self.downsample is not None:

            residual = self.downsample(
                residual
            )


        out = out + residual

        out = self.relu(out)

        return out



# ==================================================
# TCN Risk Predictor
# ==================================================

class TCNRiskPredictor(nn.Module):

    def __init__(
        self,
        input_dim=6,
        hidden_dim=64,
        num_classes=2,
        dropout=0.2
    ):

        super().__init__()


        # ------------------------------------------
        # Temporal Blocks
        # ------------------------------------------

        self.block1 = TemporalBlock(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            dilation=1,
            dropout=dropout
        )


        self.block2 = TemporalBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            dilation=2,
            dropout=dropout
        )


        self.block3 = TemporalBlock(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            dilation=4,
            dropout=dropout
        )



        # ------------------------------------------
        # Attention
        # ------------------------------------------

        self.attention = AttentionBlock(
            hidden_dim
        )



        # ------------------------------------------
        # Classifier
        # ------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_dim,
                hidden_dim // 2
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim // 2,
                num_classes
            )
        )



    def forward(self, x):

        # x shape:
        # [batch, seq_len, features]

        x = x.permute(0, 2, 1)



        # ------------------------------------------
        # Temporal Blocks
        # ------------------------------------------

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)



        # ------------------------------------------
        # Convert Back
        # ------------------------------------------

        x = x.permute(0, 2, 1)



        # ------------------------------------------
        # Attention
        # ------------------------------------------

        x = self.attention(x)



        # ------------------------------------------
        # Classification
        # ------------------------------------------

        output = self.classifier(x)

        return output
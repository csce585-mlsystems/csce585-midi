import torch
import torch.nn as nn

class DiscriminatorMLP(nn.Module):
    def __init__(self, pitch_dim, context_measures=4, hidden_sizes=[512,256], dropout=0.2, pool="concat"):
        """A simple MLP-based discriminator model.
        pitch_dim: Number of unique pitch tokens
        context_measures: Number of measures in the input sequence
        hidden_sizes: List of hidden layer sizes
        dropout: dropout is a regularization technique to prevent overfitting by randomly setting a 
            fraction of input units to 0 during training

        pool: Pooling strategy (e.g., "concat", "mean") (pooling is basically flattening the input)
            "concat" - concatenate all measures into one long vector
            "mean" - average over measures (loses contextual info)
            "max" - max over measures (loses contextual info and just shows highest activations)
        """

        # hidden sizes is a list of sizes for each hidden layer

        super().__init__()
        self.pitch_dim = pitch_dim
        self.context_measures = context_measures
        self.pool = pool

        # Determine input size based on pooling strategy
        if pool == "concat":
            input_size = pitch_dim * context_measures  # M * P from forward function
        elif pool in ["mean", "max"]:
            input_size = pitch_dim
        else:
            raise ValueError(f"Unknown pooling strategy: {pool}\n Use 'concat', 'mean', or 'max'.")
        
        # build the MLP layers
        layers = []
        # previous layer size
        prev = input_size

        # iterate over hidden sizes to create layers
        for h in hidden_sizes:
            # create a linear layer, followed by ReLU and Dropout
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h

        # final output layer (binary classification)
        layers.append(nn.Linear(prev, pitch_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # x is (B, M, P) - Batch, Measures, Pitch_dim
        if self.pool == "concat":
            B, M, P = x.shape  # Batch, Measures, Pitch_dim
            x_in = x.view(B, M * P)  # flatten measures and pitch_dim
        elif self.pool == "mean":
            x_in = x.mean(dim=1)  # average over measures
        elif self.pool == "max":
            x_in, x = x.max(dim=1)  # max over measures
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pool}\n Use 'concat', 'mean', or 'max'.")
        
        return self.net(x_in.float()) # logits for each pitch class
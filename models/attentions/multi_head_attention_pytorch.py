import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_decoder_attention_pytorch import Attention

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=2, row_dim=0, col_dim=1, num_heads=1):
        super().__init__()

        self.heads = nn.ModuleList(
            [Attention(d_model, row_dim, col_dim) for _ in range(num_heads)]
        )

        self.col_dim = col_dim

    def forward(self,
                encodings_for_q,
                encodings_for_k,
                encodings_for_v,
                mask=None # mask is a matrix with true/false
                ):
        
        return torch.cat([head(encoding_for_q,
                               encoding_for_k,
                               encoding_for_v)
                            for head in self.heads], dim=self.col_dim)

if __name__ == "__main__":
    encoding_for_q = torch.tensor(
        [[1.16, 0.23],
        [0.57, 1.36],
        [4.41, -2.16]]
    )
    encoding_for_k = torch.tensor(
        [[1.16, 0.23],
        [0.57, 1.36],
        [4.41, -2.16]]
    )
    encoding_for_v = torch.tensor(
        [[1.16, 0.23],
        [0.57, 1.36],
        [4.41, -2.16]]
    )

     # to make sure the result is always the same
    torch.manual_seed(42)

    attention = MultiHeadAttention(d_model=2, row_dim=0, col_dim=1, num_heads=2)

    result = attention(
        encoding_for_q,
        encoding_for_k,
        encoding_for_v
        )
    print(result)
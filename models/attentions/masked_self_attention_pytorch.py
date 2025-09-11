import torch
import torch.nn as nn
import torch.nn.functional as F
from encoder_decoder_attention_pytorch import Attention

class MaskedSelfAttention(nn.Module):

    def __init__(self, d_model=2, row_dim=0, col_dim=1, num_heads=1):
        super().__init__()

        self.heads = nn.ModuleList(
            [Attention(d_model, row_dim, col_dim) for _ in range(num_heads)]
        )
        
        self.col_dim = col_dim

    # mask is a matrix with true/false
    def forward(self, token_encodings, mask=None):
        q = self.W_q(token_encodings)
        k = self.W_k(token_encodings)
        v = self.W_v(token_encodings)

        sims = torch.matmul(q, k.transpose(dim0=self.row_dim, dim1=self.col_dim))
        scaled_sims = sims / torch.tensor(k.size(self.col_dim)**0.5)
        
        if mask is not None:
            # masked_fill replaces the Trues with value, and Falses as 0
            scaled_sims = scaled_sims.masked_fill(mask=mask, value=-1e9)

        attention_percents = F.softmax(scaled_sims, dim=self.col_dim)
        attention_scores = torch.matmul(attention_percents, v)

        return attention_scores
    
if __name__ == "__main__":
    encoding_matrix = torch.tensor(
        [[1.16, 0.23],
        [0.57, 1.36],
        [4.41, -2.16]]
    )

     # to make sure the result is always the same
    torch.manual_seed(42)

    maskedSelfAttention = MaskedSelfAttention(d_model=2, row_dim=0, col_dim=1)

    mask = torch.tril(torch.ones(3, 3))
    mask = mask == 0 # convert 0 to true and 1 to false
    print(mask)

    result = maskedSelfAttention(encoding_matrix, mask)
    print(result)
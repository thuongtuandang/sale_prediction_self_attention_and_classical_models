import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, heads, mask = True):
        super().__init__()
        self.input_size = input_size
        self.mask = mask
        self.heads = heads
        self.head_dim = input_size//heads
        assert(
            self.head_dim * heads == input_size
        ), "Input size needs to be divisible by heads"
        self.Wq = nn.Linear(self.input_size, self.heads * self.head_dim, bias = False)
        self.Wk = nn.Linear(self.input_size, self.heads * self.head_dim, bias = False)
        self.Wv = nn.Linear(self.input_size, self.heads * self.head_dim, bias = False)
        self.fc_out = nn.Linear(self.head_dim * heads, input_size)
        self.dropout = nn.Dropout(p = 0.1)

    def create_mask(self, query_len, key_len):
            # We need to create a matrix for mask
            # The upper part are all -inf, the lower part are all 1
            # and the digonal will be filled with 0
            lower_matrix = torch.ones(query_len, key_len)
            lower_matrix.fill_diagonal_(0)

            upper_matrix = torch.full((query_len, key_len), float('-inf'))
            upper_matrix = torch.triu(upper_matrix)
            upper_matrix.fill_diagonal_(0)

            mask = lower_matrix + upper_matrix
            return mask

    def forward(self, input):
        
        query = self.Wq(input)
        key = self.Wk(input)
        value = self.Wv(input)
        # N is the batch size
        N = input.shape[0]
        value_len, key_len, query_len = value.shape[0], key.shape[0], query.shape[0]
        # Note that each matrix in each head is of shape (value_len, head_dim)
        values = value.reshape(N, value_len, self.heads, self.head_dim)
        keys = key.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        # Compute the attention score
        # In the einsum, we keep two dimensions: N, h
        # and perform the multiplication of matrices QK^T = (len, head_dim) x (head_dim, len)
        energy = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        if self.mask:
            # Mask_fill_ method in pytorch is a little bit strange
            # for example if we have a matrix = [[1, 2, 3, 4]]
            # and we want to replace 3 by 0
            # then we need to define a mask matrix m = [[1, 1, 0, 1]]
            # note the position of 0, it is the same is 3
            # now we apply matrix.mask_fill_(m == 0, float('0.0'))
            # This will check all positions in m with values 0
            # and replace the corresponding positions in matrix with 0
             mask = self.create_mask(query_len, key_len)
             mask.unsqueeze(0).unsqueeze(0)
             energy = energy.masked_fill_(mask == float('-inf'), float('-inf'))
             energy = energy.masked_fill_(mask == 0, float('0.0'))

        attention_score = torch.nn.functional.softmax(energy/self.input_size**(1/2), dim = 3)
        # Softmax function does not change shape
        # and the attention matrix is of the shape (len, len) (two last dim)
        # and the ein sum perform the multiplication attention x V = (len, len) x (len, head_dim)
        weighted_sum = torch.einsum('nhql, nlhd -> nqhd', [attention_score, values]).reshape(
             N, query_len, self.heads*self.head_dim
             )
        output = self.fc_out(weighted_sum)
        return output

            


        
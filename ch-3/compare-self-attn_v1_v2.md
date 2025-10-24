## COMPARING SELFATTENTION_V1 AND SELFATTENTION_V2
- Note that nn.Linear in `SelfAttention_v2` uses a different weight initialization
scheme as `nn.Parameter(torch.rand(d_in, d_out))` used in `SelfAttention_v1`,
which causes both mechanisms to produce different results. 
- To check that both
implementations, `SelfAttention_v1` and `SelfAttention_v2`, are otherwise similar,
we can transfer the weight matrices from a `SelfAttention_v2` object to a
`SelfAttention_v1`, such that both objects then produce the same results.
- Your task is to correctly assign the weights from an instance of `SelfAttention_v2` to
an instance of `SelfAttention_v1`. To do this, you need to understand the
relationship between the weights in both versions. (Hint: nn.Linear stores the
weight matrix in a transposed form.) After the assignment, you should observe that
both instances produce the same outputs.

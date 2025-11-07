## GPT-2 VERSUS GPT-3
> Note that we are focusing on GPT-2 because OpenAI has made the weights of the
pretrained model publicly available, which we will load into our implementation in
chapter 6. GPT-3 is fundamentally the same in terms of model architecture, except
that it is scaled up from 1.5 billion parameters in GPT-2 to 175 billion parameters in
GPT-3, and it is trained on more data. As of this writing, the weights for GPT-3 are
not publicly available. GPT-2 is also a better choice for learning how to implement
LLMs, as it can be run on a single laptop computer, whereas GPT-3 requires a GPU
cluster for training and inference. According to Lambda Labs, it would take 355 years
to train GPT-3 on a single V100 datacenter GPU, and 665 years on a consumer RTX
8000 GPU.

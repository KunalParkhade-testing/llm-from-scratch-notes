## INFORMATION LEAKAGE

When we apply a mask and then renormalize the attention weights, it might initially
appear that information from future tokens (which we intend to mask) could still
influence the current token because their values are part of the softmax calculation.
However, the key insight is that when we renormalize the attention weights after
masking, what we're essentially doing is recalculating the softmax over a smaller
subset (since masked positions don't contribute to the softmax value).
<br>
The mathematical elegance of softmax is that despite initially including all positions
in the denominator, after masking and renormalizing, the effect of the masked
positions is nullified — they don't contribute to the softmax score in any meaningful
way.
<br>
In simpler terms, after masking and renormalization, the distribution of attention
weights is as if it was calculated only among the unmasked positions to begin with.
This ensures there's no information leakage from future (or otherwise masked)
tokens as we intended.

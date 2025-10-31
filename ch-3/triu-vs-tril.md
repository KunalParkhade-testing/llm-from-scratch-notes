In **PyTorch**, both `torch.tril()` and `torch.triu()` are used to extract parts of a **2D tensor (matrix)** — specifically, the **lower** or **upper** triangular portions.

Here’s the difference:

---

### 🧩 `torch.tril(input, diagonal=0)`

* **Meaning:** Returns the **lower triangular** part of the matrix.
* Elements **above** the specified diagonal are set to zero.

#### Example:

```python
import torch

x = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])

torch.tril(x)
```

**Output:**

```
tensor([
    [1, 0, 0],
    [4, 5, 0],
    [7, 8, 9]
])
```

So only the **lower triangle (including the main diagonal)** remains.

---

### 🧩 `torch.triu(input, diagonal=0)`

* **Meaning:** Returns the **upper triangular** part of the matrix.
* Elements **below** the specified diagonal are set to zero.

#### Example:

```python
torch.triu(x)
```

**Output:**

```
tensor([
    [1, 2, 3],
    [0, 5, 6],
    [0, 0, 9]
])
```

---

### ⚙️ Optional Parameter — `diagonal`

* You can shift which diagonal is considered the "main" one.

  * `diagonal=0`: main diagonal (default)
  * `diagonal=1`: one above the main diagonal
  * `diagonal=-1`: one below the main diagonal

#### Example:

```python
torch.tril(x, diagonal=-1)
```

Keeps only elements **below** the main diagonal.

```
tensor([
    [0, 0, 0],
    [4, 0, 0],
    [7, 8, 0]
])
```

---

### ✅ In short:

| Function     | Keeps                           | Zeros out      |
| ------------ | ------------------------------- | -------------- |
| `torch.tril` | Lower triangle (below diagonal) | Upper triangle |
| `torch.triu` | Upper triangle (above diagonal) | Lower triangle |

---

Would you like me to explain how these are used in **transformer attention masks** (a common LLM use case)?
```python
print(attn_scores)
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
print(mask)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)
```
`
tensor([[ 0.2899,  0.0716,  0.0760, -0.0138,  0.1344, -0.0511],
        [ 0.4656,  0.1723,  0.1751,  0.0259,  0.1771,  0.0085],
        [ 0.4594,  0.1703,  0.1731,  0.0259,  0.1745,  0.0090],
        [ 0.2642,  0.1024,  0.1036,  0.0186,  0.0973,  0.0122],
        [ 0.2183,  0.0874,  0.0882,  0.0177,  0.0786,  0.0144],
        [ 0.3408,  0.1270,  0.1290,  0.0198,  0.1290,  0.0078]],
       grad_fn=<MmBackward0>)
tensor([[0., 1., 1., 1., 1., 1.],
        [0., 0., 1., 1., 1., 1.],
        [0., 0., 0., 1., 1., 1.],
        [0., 0., 0., 0., 1., 1.],
        [0., 0., 0., 0., 0., 1.],
        [0., 0., 0., 0., 0., 0.]])
tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
        [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
        [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
        [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
        [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
        [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
       grad_fn=<MaskedFillBackward0>)
`

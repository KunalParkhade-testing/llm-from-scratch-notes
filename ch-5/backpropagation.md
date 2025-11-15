# 🧠 **Backpropagation — The Simple Version**

When your model predicts the next token using **softmax**, it gives a probability for every possible token.

But we want **one specific token** (the correct next token) to have the **highest probability**.

So how do we make the model do that?

## ✅ Step 1: Compare Model Prediction vs. Correct Answer

We use a **loss function** (usually cross-entropy) to measure:

> *“How wrong was the model’s prediction?”*

If the model gave a high probability to the wrong token → **high loss**
If the model gave a high probability to the correct token → **low loss**

---

## ✅ Step 2: Backpropagation

Backpropagation looks at that loss and tells every weight in the model:

> *“Hey, you contributed to the error — change by this amount to improve next time.”*

It's like giving detailed feedback to every part of the network.

---

## ✅ Step 3: Weight Update

The model adjusts its weights a tiny bit so that:

* the correct token’s probability goes **up**
* wrong token probabilities go **down**

Do this thousands or millions of times…
and the model **learns** to predict the right tokens.

---

# 🎯 **In one sentence:**

Backpropagation continuously pushes the model’s weights so that the **correct token gets a higher softmax probability** in the future.

---

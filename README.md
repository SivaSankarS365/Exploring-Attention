
# 🔥 Exploring the Cornerstone of LLMs: Attention Mechanisms

## Why This Project?

Reading papers and analyzing code is enlightening, but nothing beats the experience of getting your hands dirty, making mistakes, and learning from them. With minimal external resources, I embarked on a journey to code the cornerstone of Large Language Models (LLMs): the attention mechanism. The process was an eye-opening experience, teaching me invaluable lessons along the way.

## 🚀 Self-Attention: The Foundation of Modern NLP

Self-Attention revolutionized Natural Language Processing by drawing inspiration from retrieval systems through Query, Key, and Value. The core idea is straightforward, yet its implications are profound.

### The Formula
At the heart of self-attention lies a simple yet powerful formula:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{D}}\right) V $$

### Why Divide by  $\sqrt{D}$ ?
The division by the square root of the dimensionality \(D\) stabilizes the gradients, ensuring that the inner product values remain within a manageable range during training.

### Looking Back, Not Forward
In autoregressive models like LLMs, the model is designed to only look backward, never forward. This is crucial because the model should not see the future—only the past influences the next word generation. This constraint adds to the mystique and complexity of these models, mimicking the way humans process language.

## 🐉 The Real Beast: Multi-Headed Attention

Multi-Headed Attention is where the true power of LLMs shines. By enabling parallelism, multiple queries can attend to various aspects of the text simultaneously, allowing the model to capture diverse patterns and meanings in a single pass.

### Grouped Query Attention: Reining the Beast

To further optimize and reduce compute latency, grouped query attention steps in. This technique clusters queries, reducing the computational overhead while maintaining the model's ability to attend to multiple aspects of the input efficiently.

## ⚙️ Features

- **Modular Components**: The codebase is highly modular, allowing customization through inheritance.
- **Built for Learning**: The code is simple and straightforward, designed for those who want to understand the intricacies of attention mechanisms.
- **LLM Tokenizer Insights**: Did you know that pad tokens must be prepended in LLM tokenizers? This is just one of the many insights I gained while working on this project.
- **Tutorial Notebook**: Accompanying the code is a `Tutorial.ipynb` notebook that walks through the step-by-step execution and reveals the underlying intricacies of the model.
- **Comprehensive Implementation**: The project includes implementations of self-attention, multi-headed attention, and grouped query attention, complete with KV-caching.
- **Masking Explained**: Detailed explanations of causal masks and pad-attention masks are provided to deepen your understanding.

## 🛠️ Components

- **Self-Attention**: The backbone of modern NLP models.
- **Multi-Headed Attention**: Parallelized attention across multiple heads for richer representations.
- **Grouped Query Attention**: Optimized attention with reduced compute latency.
- **More**: In future:)

## 💡 Behind the Hood

Dive deep into the various attention mechanisms that power LLMs. This project is your gateway to understanding self-attention, multi-headed attention, and grouped query attention, all with KV-caching.

---

Happy Coding! 😊

---

Feel free to customize it further to match your style and needs!

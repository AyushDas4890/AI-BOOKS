# Chapter 0: Pipeline Fixture

## Math
Inline math $\hat{y} = \sigma(w^T x + b)$ in a sentence.

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]$$

## Code
```python
import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
print(sigmoid(np.array([0.0, 2.0])))
```

## Table
| Metric | Formula | Use |
|---|---|---|
| Precision | $\frac{TP}{TP+FP}$ | Costly false positives |

> **Interview tip:** always state assumptions first.

<div class="qa"><p class="q">Q: Why sigmoid for logistic regression?</p>
<p>Because it maps log-odds to (0,1)...</p></div>

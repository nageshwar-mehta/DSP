The **Cramér-Rao Lower Bound (CRLB)** provides a theoretical lower bound on the variance of any unbiased estimator of a deterministic parameter. It is crucial in signal processing and estimation theory, especially in evaluating the performance limits of localization algorithms.

---

### **1. CRLB: Definition and Basic Formulation**

Let:

* $\boldsymbol{\theta} \in \mathbb{R}^k$: vector of parameters to be estimated (e.g., source location)
* $\hat{\boldsymbol{\theta}}$: an unbiased estimator of $\boldsymbol{\theta}$
* $\mathcal{I}(\boldsymbol{\theta})$: Fisher Information Matrix (FIM)

Then, the CRLB is given by:

$$
\text{Cov}(\hat{\boldsymbol{\theta}}) \succeq \mathcal{I}^{-1}(\boldsymbol{\theta})
$$

This means that the covariance matrix of any unbiased estimator is greater than or equal to the inverse of the Fisher Information Matrix in the positive semi-definite sense.

---

### **2. Fisher Information Matrix (FIM)**

For a likelihood function $\mathcal{L}(\boldsymbol{\theta}) = p(\mathbf{x}; \boldsymbol{\theta})$, the FIM is defined as:

$$
\mathcal{I}_{ij}(\boldsymbol{\theta}) = \mathbb{E} \left[ \left( \frac{\partial \ln \mathcal{L}(\boldsymbol{\theta})}{\partial \theta_i} \right) \left( \frac{\partial \ln \mathcal{L}(\boldsymbol{\theta})}{\partial \theta_j} \right) \right]
$$

Or, equivalently,

$$
\mathcal{I}(\boldsymbol{\theta}) = - \mathbb{E} \left[ \frac{\partial^2 \ln \mathcal{L}(\boldsymbol{\theta})}{\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}^T} \right]
$$

---

### **3. CRLB for Source Localization (e.g., via TOA, AOA, or RSS)**

Let us consider **TOA-based source localization** for simplicity, with the measurement model:

$$
t_i = \frac{\|\mathbf{x} - \mathbf{s}_i\|}{c} + n_i
$$

Where:

* $\mathbf{x}$: source location (unknown parameter)
* $\mathbf{s}_i$: known sensor location
* $c$: propagation speed
* $n_i \sim \mathcal{N}(0, \sigma^2)$: Gaussian noise

The log-likelihood of observations $\mathbf{t} = [t_1, \dots, t_M]^T$ is:

$$
\ln \mathcal{L}(\mathbf{x}) = -\frac{1}{2\sigma^2} \sum_{i=1}^M \left( t_i - \frac{\|\mathbf{x} - \mathbf{s}_i\|}{c} \right)^2 + \text{const}
$$

Take the gradient of the log-likelihood with respect to $\mathbf{x}$:

$$
\frac{\partial \ln \mathcal{L}}{\partial \mathbf{x}} = \frac{1}{\sigma^2 c} \sum_{i=1}^M \left( t_i - \frac{\|\mathbf{x} - \mathbf{s}_i\|}{c} \right) \cdot \frac{\mathbf{x} - \mathbf{s}_i}{\|\mathbf{x} - \mathbf{s}_i\|}
$$

From this, compute the **FIM** and finally the CRLB:

$$
\text{CRLB}(\mathbf{x}) = \left( \mathcal{I}(\mathbf{x}) \right)^{-1}
$$

This gives a **lower bound** on the variance of any unbiased estimate of the source location.

---

### **4. Interpretation**

* The **diagonal entries** of the CRLB give the lower bounds on the variance of the individual parameters.
* If a proposed estimator (like ML or a deep-learning-based method) achieves this bound, it is said to be **efficient**.

---

### **5. Application in our Paper**

In our reference paper, the CRLB is discussed as a benchmark to **evaluate the accuracy** of the proposed localization scheme. Specifically, the authors derive or refer to the CRLB under their system model and compare it with simulation results to show how close their proposed method is to the theoretical bound, which confirms the estimator's efficiency or near-optimality .



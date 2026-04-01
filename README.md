<div id="top">

<!-- HEADER STYLE: CLASSIC -->
<div align="center">


# SOCP-LP-SOLVER



</div>
<br>

---

## Table of Contents

- [Overview](#overview)
- [Mathematical Model](#mathematical-model)
- [Algorithm](#algorithm)
- [Example](#example)
- [Getting Started](#getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)

---

## Overview

socp-lp-solver is a Python package that provides advanced solvers for Support Vector Machine (SVM) optimization problems using Second-Order Cone Programming (SOCP). Designed for high-dimensional and large-scale data, it enables efficient, robust with respecto to noise and interpretable machine learning models.

**Authors:** Miguel Carrasco, Julio Lopez, Matthieu Marechal

**Why svm-socp-lp-solvers?**

This project aims to simplify and accelerate the development of SVM-based solutions with cutting-edge convex optimization techniques. The core features include:

- 🧩 **🔧 Modular Architecture:** Seamlessly integrates with existing ML workflows and scales across modules.
- 🚀 **⚙️ High-Performance Solvers:** Efficiently handles large-scale SOCP problems for support vector machines.
- 📊 **📈 Utility Functions:** Provides tools for model inference, including predictions and probability estimates.
- 🧮 **🔍 Focused on Convex Optimization:** Leverages cvxpy and other libraries for robust, reliable solutions, despite the fact that the original problem is not convex.
- 🧠 **🤖 Support for Sparse, Robust Models:** Facilitates feature selection and high-dimensional data handling, robust with respect to noise.

---

## Mathematical model 

### SOCP_Lp

This estimator solves the following optimization problem:

$$
\min_{w,b,\xi}\ \sum_{j=1}^n (|w_j|+\varepsilon)^p + C \xi
\quad \mathrm{s.t.}\quad
\begin{aligned}
& (w,b,\xi) \in \mathbb{R}^{n+2} \\
& w^\top \mu_1 + b \ge 1 - \xi + \kappa(\alpha_1)\|S_1^\top w\|, \\
& -(w^\top \mu_2 + b) \ge 1 - \xi + \kappa(\alpha_2)\|S_2^\top w\|, \\
& \xi \ge 0.
\end{aligned}
$$

where $\kappa(\alpha_i)=\sqrt{\frac{\alpha_i}{1-\alpha_i}}$. The vector $\mu_1$ (resp. $\mu_2$) is the mean feature vector associated with the
positive (resp. negative) class.

The matrix $S_j \in \mathbb{R}^{n \times m_j}$, with $j \in \{1,2\}$, satisfies
$\Sigma_j = S_j S_j^\top$, where $\Sigma_1$ (resp. $\Sigma_2$) is the covariance
matrix of the features associated with the positive (resp. negative) class.

The constraint set above is a reformulation of the following probabilistic
constraint using the multivariate Chebyshev inequality:

$$
\inf_{\tilde{x}_j \sim (\mu_j,\Sigma_j)}
\Pr\left\( (-1)^{j+1}(w^\top \tilde{x}_j + b) \ge 0 \right\)
\ge \alpha_j, \quad j = 1,2.
$$

The notation $\tilde{x}_j \sim (\mu_j,\Sigma_j)$ indicates that the random vectors
$\tilde{x}_j$ have mean $\mu_j$ and covariance matrix $\Sigma_j$.

This model can be interpreted as a robust version of SVM_Lp.

The smoothing parameter $\varepsilon > 0$ makes the objective locally Lipschitz
and avoids singular behavior at $w_j = 0$.

## Algorithm

IRL1 for the $\ell_p$-XiSOCP Model ($0 < p < 1$)

**Input:** <br>
    Training data $(X, y)$<br>
    Parameters: $p, c, \varepsilon > 0$, $\kappa = (\kappa_1, \kappa_2)$<br>
    Maximum iterations: max_iter<br>
    Tolerance: tol

**Preprocessing:** <br>
    Split data into classes A ($y=+1$) and B ($y=-1$)<br>
    Compute class means $\mu_1$, $\mu_2$<br>
    Compute matrices $S_1$, $S_2$:<br>
        - either via Cholesky of covariance matrices, or<br>
        - via sample-based estimation

**Initialize:** <br>
$k = 0$ <br>
$Φ^{0} = 1$  (vector of ones), <br>
$w^{0}$ arbitrary (e.g., constant vector).

**Repeat:** <br>
**Step 1:** Solve weighted SOCP subproblem via CVX <br>
minimize   $\|\Phi\otimes w\|_1 + C \xi$<br>
subject to<br>
$\kappa_1 \|S_1^T w\| \leq w^\top \mu_1 + b - 1 + \xi$<br>
$\kappa_2 \|S_2^T w\| \leq w^\top \mu_2 + b - 1 + \xi$<br>
$\xi\geq0$

**Step 2:** Update IRL1 weights<br>
$\Phi_i^{k+1}=\dfrac{p}{(|w_i^k|+\varepsilon)^{1-p}}$, for $i=1,\cdots,n$<br>

**Step 3:** Check convergence<br>
if $\|w^{k+1} - w^k\|_\infty < tol$ or k+1>max_iter <br>
stop

**Step 4:** $k \leftarrow k + 1$

**Output:** <br>
Final solution $(w, b,\xi)$<br>
Sparse feature set: indices where $|w_i| > threshold$

## Example


    from socp_lp_solver import SOCP_Lp
    import pandas as pd
    
    url = "https://raw.githubusercontent.com/mmatthieu1290/svm-socp-lp-solvers/main/datos_Titanic.xlsx"
    df = pd.read_excel(url, engine="openpyxl")
    X = df.iloc[:,:-1]
    y = df.iloc[:,-1]

    socp = SOCP_Lp(p=0.1,alpha_1=0.2,alpha_2=0.2)
    socp.fit(X,y)

    print("Coefs : ",socp.coef_)
    print("Selected features : ",socp.selected_feature_names_)

## Getting Started

### Prerequisites

This project requires the following dependencies:

- **Programming Language:** Python
- **Package Manager:** Conda

### Installation

1. **Install the library:**

    ```sh
    pip install git+https://github.com/mmatthieu1290/socp-lp-solver.git
    ```
2. **Import the solver:**

    ```sh
    from socp_lp_solver import SOCP_Lp
    ```








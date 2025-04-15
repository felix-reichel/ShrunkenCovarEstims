import numpy as np
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, OAS, ShrunkCovariance, EmpiricalCovariance
from numpy.linalg import norm

np.random.seed(123)

# Params
p = 20
true_cov = np.eye(p)  # true covar mat
n_values = [5, 10, 20, 50, 100]  # n
num_trials = 500  # sim trials


mse_standard = []
mse_modified = []
mse_ledoit = []
mse_oas = []
mse_shrunk = []
mse_empirical = []

for n in n_values:
    err_std_list = []
    err_mod_list = []
    err_ledoit_list = []
    err_oas_list = []
    err_shrunk_list = []
    err_empirical_list = []

    for _ in range(num_trials):
        X = np.random.multivariate_normal(np.zeros(p), true_cov, size=n)
        x_bar = X.mean(axis=0, keepdims=True)

        # Standard sample covar (unbiased)
        Q_std = (1 / (n - 1)) * ((X - x_bar).T @ (X - x_bar))

        # Modified "Bariance" covar biased estimator with n + 1 in denominator
        Q_mod = (1 / (n + 1)) * ((X - x_bar).T @ (X - x_bar))

        # Ledoit-Wolf estimator
        Q_lw = LedoitWolf().fit(X).covariance_

        # Oracle Approximating Shrinkage (OAS)
        Q_oas = OAS().fit(X).covariance_

        # Shrunk covariance with default shrinkage
        Q_shrunk = ShrunkCovariance().fit(X).covariance_

        # Empirical covariance from sklearn
        Q_emp = EmpiricalCovariance().fit(X).covariance_

        # Compute squared Frobenius norm errors
        err_std_list.append(norm(Q_std - true_cov, 'fro')**2)
        err_mod_list.append(norm(Q_mod - true_cov, 'fro')**2)
        err_ledoit_list.append(norm(Q_lw - true_cov, 'fro')**2)
        err_oas_list.append(norm(Q_oas - true_cov, 'fro')**2)
        err_shrunk_list.append(norm(Q_shrunk - true_cov, 'fro')**2)
        err_empirical_list.append(norm(Q_emp - true_cov, 'fro')**2)

    mse_standard.append(np.mean(err_std_list))
    mse_modified.append(np.mean(err_mod_list))
    mse_ledoit.append(np.mean(err_ledoit_list))
    mse_oas.append(np.mean(err_oas_list))
    mse_shrunk.append(np.mean(err_shrunk_list))
    mse_empirical.append(np.mean(err_empirical_list))

plt.figure(figsize=(12, 7))
plt.plot(n_values, mse_standard, marker='o', label='Standard (n - 1)')
plt.plot(n_values, mse_modified, marker='s', label='Modified "Bariance" (n + 1)')
plt.plot(n_values, mse_ledoit, marker='^', label='Ledoit-Wolf')
plt.plot(n_values, mse_oas, marker='x', label='OAS')
plt.plot(n_values, mse_shrunk, marker='*', label='Shrunk Covariance')
plt.plot(n_values, mse_empirical, marker='d', label='EmpiricalCovariance')
plt.xlabel('Sample Size (n)')
plt.ylabel('MSE of Covariance Estimate')
plt.title('Covariance Estimator MSE Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# CANOPY-TD formulas for MS Word Equation Editor

R_i = R_0 + alpha * rho_i

rho_i = sum_(j != i) I(d_ij < R_0)

d_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)

O_ij = max(0, R_i + R_j - d_ij)

O_total = sum_i sum_(j > i) O_ij

E_i = sum_(j != i) O_ij

T_i^(t+1) = T_i^t - beta * grad(E_i)

T_i^(t+1) = T_i^t - beta * grad(sum_(j != i) O_ij) + gamma * N_i

C = sum_i sum_(j > i) I(d_ij < d_safe)

Risk = sum_i sum_(j > i) rho_ij / (d_ij + epsilon)

D_mean = (1/N) * sum_i ||T_i^final - T_i^initial||

F = lambda_1 * O_total + lambda_2 * C + lambda_3 * Risk + lambda_4 * D_mean

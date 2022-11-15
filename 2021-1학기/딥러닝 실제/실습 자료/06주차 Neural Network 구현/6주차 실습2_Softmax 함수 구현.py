import numpy as np

a = np.array([0.3,2.9,4.0])
exp_a = np.exp(a)
print(exp_a)

sum_exp = np.sum(exp_a)
print(sum_exp)

y = exp_a / sum_exp
print(y)
print(np.sum(y))
import numpy as np

def conjugate_direction(func, grad, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    grad_val = np.array([g(*x) for g in grad])
    d = -grad_val
    for _ in range(max_iter):
        if np.linalg.norm(grad_val) < tol:
            return x, _
        alpha = 1e-2  # Pas fixe
        x += alpha * d
        grad_new = np.array([g(*x) for g in grad])
        beta = np.dot(grad_new, grad_new) / np.dot(grad_val, grad_val)
        d = -grad_new + beta * d
        grad_val = grad_new
    return x, max_iter
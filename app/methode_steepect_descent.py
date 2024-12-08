import numpy as np

# Méthodes d'optimisation
def steepest_descent(func, grad, x0, tol=1e-6, max_iter=100):
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(max_iter):
        grad_val = np.array([g(*x) for g in grad])
        if np.linalg.norm(grad_val) < tol:
            return x, _, trajectory
        step_size = 1e-2
        x -= step_size * grad_val
        trajectory.append(x.copy())
    return x, max_iter, trajectory

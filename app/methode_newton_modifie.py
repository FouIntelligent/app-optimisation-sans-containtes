import numpy as np

def newton_modified(func, grad, hess, x0, tol=1, max_iter=100):
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(max_iter):
        grad_val = np.array([g(*x) for g in grad])
        hess_val = np.array([[hij(*x) for hij in row] for row in hess])
        if np.linalg.norm(grad_val) < tol:
            return x, _,trajectory
        hess_val += np.eye(len(x)) * 1e-3  # Régularisation
        try:
            delta_x = np.linalg.solve(hess_val, -grad_val)
        except np.linalg.LinAlgError:
            return x, _, trajectory
        x += delta_x
        trajectory.append(x.copy())
    return x, max_iter
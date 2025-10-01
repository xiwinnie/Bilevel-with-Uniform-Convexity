# deterministic tuning
import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(42)
noise_rate = 1

def project_to_ball(w, center, radius):
    """Projects w onto a ball centered at `center` with radius `radius`."""
    norm_diff = np.linalg.norm(w - center)
    if norm_diff > radius:
        return center + radius * (w - center) / norm_diff
    return w


def epoch_sgd(gamma1, T1, D1, T, p, x):
    """Epoch-SGD optimization algorithm."""
    tau = 2 * (p - 1) / p
    k = 1
    w1 = np.random.randn()

    while sum([T1 * (2 ** (tau * i)) for i in range(k)]) <= T and k < 50:
        Tk = int(T1 * (2 ** (tau * (k - 1))))
        Dk = D1 / (2 ** ((k - 1) / p))
        gamma_k = gamma1 / (2 ** (k - 1))
        wk = np.copy(w1)
        W = []
        for t in range(Tk):
            print(f'{x} {wk}')
            noise = np.random.normal(0, noise_rate)
            grad = grad_g(x, wk, p) + noise
            wk = project_to_ball(wk - gamma_k * grad, w1, Dk)
            # w1 = w1 - gamma_k * grad
            W.append(wk)
            # print(f'{t}:{wk}')
        w1 = np.mean(W)

        k += 1

    return w1


def f(y, p):
    """Upper-level objective function."""
    y_threshold = (math.pi / 2) ** (1 / (p - 1))
    if y > y_threshold:
        return 1
    elif -y_threshold <= y <= y_threshold:
        return math.sin(y ** (p - 1))
    else:
        return -1


def g(x, y, p):
    """Lower-level objective function."""
    return y ** p / p - y * math.sin(x)


def grad_g(x, y, p):
    """Lower-level gradient function."""
    return y ** (p - 1) - math.sin(x)


def hypergradient(x, y, p):
    y_threshold = (math.pi / 2) ** (1 / (p - 1))
    if -y_threshold <= y <= y_threshold:
        return np.cos(x) * np.cos(np.sin(x))
    else:
        return 0
        # grad_g,f,x_0,y_0,p, eta, beta, alpha,a, K, T1, R1, T


def unibio(x0, y0, p, eta, beta, alpha, a, K, T1, R1, T):
    """UniBiO: Uniformly Convex Bilevel Optimization Algorithm."""
    x = x0
    y = y0
    m = 0
    grad_norms = []
    function_values = []

    for t in range(T):
        if t % a == 0:
            y = epoch_sgd(alpha, T1, R1, K, p, x)
        noise = np.random.normal(0, 0.01)
        hypergrad = hypergradient(x, y, p) + noise
        m = beta * m + (1 - beta) * hypergrad
        x = x - eta * m / (np.linalg.norm(m) + 1e-8)
        grad_norms.append(np.linalg.norm(hypergrad))
    return grad_norms


# Run experiments for different values of p
p_values = [2, 4, 6, 8]
x_0 = 1
y_0 = 1
T = 500
beta = 0.9
alpha = [1, 1, 1, 1]
eta = [0.05, 0.03, 0.02, 0.01]
a = 2
K = 100
T1 = 5
R1 = 1

# Plot convergence of hypergradient for different values of p
plt.figure(figsize=(6, 5))
plt.rcParams.update({'font.size': 14, 'font.weight': 'bold'})
colors = ['blue', 'red', 'green', 'orange']
for i, p in enumerate(p_values):
    grad_norms = unibio(x_0, y_0, p, eta[i], beta, alpha[i], a, K, T1, R1, T)
    plt.plot(range(T), grad_norms, label=f"p={p}", color=colors[i], linewidth=2.0)

plt.xlabel(f"Iterations", fontweight='bold')
plt.ylabel(r"Hypergradient Norm $\|\nabla f\|$", fontweight='bold')
plt.title("Hypergradient Norm under Different p Values", fontweight='bold', fontsize="12")
plt.legend()
plt.grid(linestyle='-.')
plt.tight_layout()
plt.savefig(f"stochastic_hyperg_noise_{noise_rate}.pdf", transparent=True)
# plt.show()
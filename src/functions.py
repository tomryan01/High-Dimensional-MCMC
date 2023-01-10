import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from scipy.stats import norm
import matplotlib.cm as cm
import copy
from tqdm import tqdm


def GaussianKernel(x, l):
    """ Generate Gaussian kernel matrix efficiently using scipy's distance matrix function"""
    D = distance_matrix(x, x)
    return np.exp(-pow(D, 2)/(2*pow(l, 2)))


def subsample(N, factor, seed=None):
    assert factor>=1, 'Subsampling factor must be greater than or equal to one.'
    N_sub = int(np.ceil(N / factor))
    if seed: np.random.seed(seed)
    idx = np.random.choice(N, size=N_sub, replace=False)  # Indexes of the randomly sampled points
    return idx


def get_G(N, idx):
    """Generate the observation matrix based on datapoint locations.
    Inputs:
        N - Length of vector to subsample
        idx - Indexes of subsampled coordinates
    Outputs:
        G - Observation matrix"""
    M = len(idx)
    G = np.zeros((M, N))
    for i in range(M):
        G[i,idx[i]] = 1
    return G


def probit(v):
    return np.array([0 if x < 0 else 1 for x in v])


def predict_t(samples):
    return np.mean(norm.cdf(samples), axis=0)


###--- Density functions ---###

def log_prior(u, K_inverse):
    n = len(u)
    _, logDetK = np.linalg.slogdet(np.linalg.inv(K_inverse + 1e-6 * np.eye(n)))
    return -0.5*(n*np.log(2*np.pi) + logDetK + (u.T @ K_inverse @ u))


def log_continuous_likelihood(u, v, G):
    n = len(v)
    return -0.5*(n*np.log(2*np.pi) + (v - G @ u).T @ (v - G @ u))


def log_probit_likelihood(u, t, G):
    phi = norm.cdf(G @ u)
    log_p_1 = np.log(phi)
    log_p_0 = np.log(np.ones(len(phi)) - phi)
    return np.dot(t, log_p_1) + np.dot(1-t, log_p_0)


def log_poisson_likelihood(u, c, G):
    u_sub = G @ u
    out = -np.sum(np.exp(u_sub))
    out += np.dot(c, u_sub)
    return out


def log_continuous_target(u, y, K_inverse, G):
    return log_prior(u, K_inverse) + log_continuous_likelihood(u, y, G)


def log_probit_target(u, t, K_inverse, G):
    return log_prior(u, K_inverse) + log_probit_likelihood(u, t, G)


def log_poisson_target(u, c, K_inverse, G):
    return log_prior(u, K_inverse) + log_poisson_likelihood(u, c, G)


###--- MCMC ---###

def grw(log_target, u0, data, K, G, n_iters, beta):
    """ Gaussian random walk Metropolis-Hastings MCMC method
        for sampling from pdf defined by log_target.
    Inputs:
        log_target - log-target density
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0
    N = K.shape[0]

    # Inverse computed before the for loop for speed
    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))
    Kc_inverse = np.linalg.inv(Kc)
    K_inverse = Kc_inverse.T @ Kc_inverse # use cholesky factor for matrix inversion for better stability

    lt_prev = log_target(u_prev, data, K_inverse, G)

    for i in tqdm(range(n_iters)):

        u_new = u_prev + beta * Kc @ np.random.randn(N) # use cholesky factor for stable sampling

        lt_new = log_target(u_new, data, K_inverse, G)

        log_alpha = min(0, lt_new-lt_prev)
        log_u = np.log(np.random.random())

        # Accept/Reject
        accept = (log_u <= log_alpha)
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            lt_prev = lt_new
        else:
            X.append(u_prev)

    return X, acc / n_iters


def pcn(log_likelihood, u0, y, K, G, n_iters, beta):
    """ pCN MCMC method for sampling from pdf defined by log_prior and log_likelihood.
    Inputs:
        log_likelihood - log-likelihood function
        u0 - initial sample
        y - observed data
        K - prior covariance
        G - observation matrix
        n_iters - number of samples
        beta - step-size parameter
    Returns:
        X - samples from target distribution
        acc/n_iters - the proportion of accepted samples"""

    X = []
    acc = 0
    u_prev = u0
    N = K.shape[0]

    ll_prev = log_likelihood(u_prev, y, G)

    Kc = np.linalg.cholesky(K + 1e-6 * np.eye(N))

    for i in tqdm(range(n_iters)):

        u_new = np.sqrt(1-beta**2)*u_prev + beta * Kc @ np.random.randn(N)

        ll_new = log_likelihood(u_new, y, G)

        log_alpha = min(0, ll_new - ll_prev)
        log_u = np.log(np.random.random())

        # Accept/Reject
        accept = (log_u <= log_alpha)
        if accept:
            acc += 1
            X.append(u_new)
            u_prev = u_new
            ll_prev = ll_new
        else:
            X.append(u_prev)

    return X, acc / n_iters


###--- Plotting ---###

def plot_3D(u, x, y, title=None):
    """Plot the latent variable field u given the list of x,y coordinates"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)
    if title:  plt.title(title)
    plt.show()


def plot_2D(counts, xi, yi, title=None, countsMin=False, colors='viridis'):
    """Visualise count data given the index lists"""
    Z = -np.ones((max(yi) + 1, max(xi) + 1))
    for i in range(len(counts)):
        Z[(yi[i], xi[i])] = counts[i]
    my_cmap = copy.copy(cm.get_cmap(colors))
    my_cmap.set_under('k', alpha=0)
    fig, ax = plt.subplots()
    if countsMin:
        im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[np.min(counts), np.max(counts)])
    else:
        im = ax.imshow(Z, origin='lower', cmap=my_cmap, clim=[-0.1, np.max(counts)])
    fig.colorbar(im)
    if title:  plt.title(title)
    plt.show()


def plot_result(u, data, x, y, x_d, y_d, title=None):
    """Plot the latent variable field u with the observations,
        using the latent variable coordinate lists x,y and the
        data coordinate lists x_d, y_d"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.plot_trisurf(x, y, u, cmap='viridis', linewidth=0, antialiased=False)
    ax.scatter(x_d, y_d, data, marker='x', color='r')
    if title:  plt.title(title)
    plt.show()

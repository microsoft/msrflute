# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
import numpy as np
from scipy.special import gammainc
from sklearn.cluster import KMeans
from sklearn import cluster as skcluster


kmeans_single = skcluster._kmeans.lloyd_iter_chunked_dense


def sample(ndim, r, num_samples=1):
    x = np.random.normal(size=(num_samples, ndim))
    ssq = np.sum(x**2,axis=1)
    fr = r*gammainc(ndim/2,ssq/2)**(1/ndim)/np.sqrt(ssq)
    if num_samples > 1:
        fr = np.tile(fr.reshape(num_samples,1),(1,ndim))
    return  np.multiply(x,fr)


def sphere_packing_initialization(n_clusters, n_dim, min_cluster_radius,
                                  max_space_size, max_failed_cases, verbose=None):
    a, max_r = min_cluster_radius, max_space_size
    centers = np.empty((n_clusters, n_dim))
    cluster_id = 0
    fail_count = 0
    r = max_r - a
    while cluster_id < n_clusters:
        v = sample(n_dim, r)
        if cluster_id > 0 and np.min(np.linalg.norm(centers[:cluster_id, :] - v, axis=-1)) < 2 * a:
            fail_count += 1
            if fail_count >= max_failed_cases:
                fail_count = 0
                cluster_id = 0
                a = a / 2 # TODO Use binary search to find maximum a that don't fail (vaguely discribed in the diff-p kmeas paper)
                if verbose:
                    print(f'Failing to pack, halving min_cluster_radius to {a}')
                r = max_r - a
            continue
     
        centers[cluster_id] = v
        cluster_id += 1
    if verbose:
        print('Final min_cluster_radius', a)
    return centers, a


def add_gaussian_noise(centers_new, weight_in_clusters, eps,
                       max_cluster_l2, max_sample_weight,
                       cluster_to_weight_ratio=-1, delta=1e-7, verbose=None):
    scaler = 1
    
    if cluster_to_weight_ratio > 0:
        # Compute the scaler to apply to the sample weights
        scaler = max_cluster_l2 / (max_sample_weight * cluster_to_weight_ratio)
    max_sample_weight *= scaler
   
    max_l2_sensitivity = np.sqrt(max_cluster_l2 ** 2 + max_sample_weight ** 2)
    sigma = np.sqrt(2 * np.log(1.25 / delta)) * max_l2_sensitivity / eps
    if verbose:
        print('cluster_to_weight_ratio', cluster_to_weight_ratio,
              'scaler', scaler,
              'max_sample_weight', max_sample_weight,
              'max_l2_sensitivity', max_l2_sensitivity,
              'sigma', sigma)
    centers_sum = (centers_new * weight_in_clusters.reshape(-1, 1)) + np.random.normal(scale=sigma, size=centers_new.shape)
    # Scale the sample weights by scaling the cluster weights, since (s*w1 + s*w2, ...) == s*(w1 + w2 + ...), where s is the scaler
    # Add noise then rescale back. We should never get negative weights because of the noise
    weight_in_clusters[:] = np.maximum(1e-10, (weight_in_clusters * scaler) + np.random.normal(scale=sigma, size=weight_in_clusters.shape)) / scaler
    centers_new[:] = centers_sum / weight_in_clusters.reshape(-1, 1)


def DPKMeans(n_dim, eps, max_cluster_l2, max_sample_weight=1.0,
             max_iter=300, cluster_to_weight_ratio=-1, n_clusters=8,
             tol=1e-4, verbose=0, delta=1e-7, max_failed_cases=300,
             min_cluster_radius=None, **kwargs):
    """Differentially private KMeans

    Initialise the differentially-private Sklearn.cluster.KMeans overriding lloyd algorithm,
    by adding Gaussian noise.

    Parameters
    ---------
    
    n_dim : int
        The dimension size of the input space
    
    eps : float
        The privacy loss (epsilon) per iteration. Currently only fix epsilon is implemented so
        the overall privacy loss <= eps * max_iter

    max_cluster_l2 : float
        The maximum l2 norm of any example vector that we want to cluster

    max_sample_weight : float
        The maximum weight of a sample default=1.0

    max_iter : int, default=300
        Maximum number of iterations of the k-means algorithm for a
        single run.

    cluster_to_weight_ratio : float, default=-1
        The ratio max_cluster_l2 / max_sample_weight used to scale the cluster counts before adding the noise
        If it is set to -1, do not scale the counts

    n_clusters : int, default=8
        The number of clusters to form as well as the number of
        centroids to generate.

    tol : float, default=1e-4
        Relative tolerance with regards to Frobenius norm of the difference
        in the cluster centers of two consecutive iterations to declare
        convergence.

    verbose : int, default=0
        Verbosity mode.

    delta : float, default=1e-7
        Gaussian mechanism delta or probability of failure, should be set < 1/num of examples

    max_failed_cases : int, default=300
        The number of sampling trails in sphere packing before halving the minimum cluster radius

    min_cluster_radius : float, default=None (= max_cluster_l2 / n_clusters)
        Half the minimum distance between clusters centers
    """

    if min_cluster_radius is None:
        min_cluster_radius = max_cluster_l2 / n_clusters

    # Initalise the cluster centers using sphere packing
    init_centers, min_cluster_radius = sphere_packing_initialization(n_clusters, n_dim,
                                                                     min_cluster_radius,
                                                                     max_cluster_l2,
                                                                     max_failed_cases,
                                                                     verbose)

    final_eps = [0] # To keep track of the actual number of iterations until convergence
    def modified_lloyd(X, sample_weight, x_squared_norms, centers, centers_new,
                       weight_in_clusters, labels, center_shift, n_threads,
                       update_centers=True):

        # Clip the maximum client contribution to the cluster count
        sample_weight = np.minimum(sample_weight, max_sample_weight)
        
        if not update_centers:
            return kmeans_single(X, sample_weight, x_squared_norms, centers, centers_new,
                                weight_in_clusters, labels, center_shift, n_threads, update_centers=False)
        
        
        # Scale input vectors if necessary
        if np.max(x_squared_norms) > max_cluster_l2 ** 2:
            if verbose:
                print(f'Scaling the input examples as their l2 norm is larger than {max_cluster_l2}')
            scaler_squared = np.minimum(max_cluster_l2 ** 2 / x_squared_norms, 1.0)
            x_squared_norms[:] = x_squared_norms * scaler_squared
            X[:] = X * np.sqrt(scaler_squared).reshape(-1, 1)
        
        kmeans_single(X, sample_weight, x_squared_norms, centers, centers_new,
                      weight_in_clusters, labels, center_shift, n_threads)

        # Add noise to centers_new
        add_gaussian_noise(centers_new, weight_in_clusters, eps,
                           max_cluster_l2, max_sample_weight,
                           cluster_to_weight_ratio, delta=delta,
                           verbose=verbose)

        # Other values need to be changed because of that: center_shift, labels, 
        center_shift[:] = np.linalg.norm(centers - centers_new, axis=-1)
        # Run E-step of kmeans to get the new labels
        kmeans_single(X, sample_weight, x_squared_norms, centers, centers_new,
                    weight_in_clusters, labels, center_shift, n_threads, update_centers=False)

        # Increment the number of iterations
        final_eps[0] += eps

    sys.modules[KMeans.__module__].lloyd_iter_chunked_dense = modified_lloyd

    kmeans = KMeans(n_clusters=n_clusters,
                    algorithm='full',
                    init=init_centers,
                    verbose=verbose,
                    max_iter=max_iter,
                    tol=tol, **kwargs)
    kmeans.eps = final_eps
    return kmeans


def resetKMeans():
    sys.modules[KMeans.__module__].lloyd_iter_chunked_dense = kmeans_single
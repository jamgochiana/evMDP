import numpy as np
from scipy import stats

class energyGMM(object):
    """
    Saves relevant information from scikit-learn GaussianMixtureModel class.
    Also can perform exact inference, mean finding, etc

    Attributes:
        weights_
        means_
        covariances_
        
        time_range

        observations
        posterior_weights
        posterior_means
        posterior_covariances


    To use:
    >>> gmm = Square(3)
    >>> sq.area
    9
    >>> sq.perimeter
    12
    >>> sq.area = 16
    >>> sq.side
    4
    >>> sq.perimeter
    16    

    """

    def __init__(self,gmm,time_range,normalize=True):
        """Initializes mixture model class."""

        # take from gmm class
        self.weights_ = gmm.weights_.copy()
        self.means_ = gmm.means_.copy()
        self.covariances_ = gmm.covariances_.copy()
        
        self.time_range = np.arange(time_range[0], time_range[1]+1)
        if len(self.time_range) != self.means_.shape[1]:
            raise ValueError('Incorrect size time range')

        # normalize price to mean MLE of 1. by default 
        if normalize:

            # calculate MLE Price and scaling factor
            P_MLE = self.means_.T.dot(self.weights_)
            scaling_factor = P_MLE.mean()

            # scale down means and covariances
            self.means_ = self.means_ / scaling_factor
            self.covariances_ = self.covariances_ / (scaling_factor**2)

        # initialize observed costs as empty np array
        self.observations = np.array([])

        # initialize posterior distribution to initial distribution
        self.posterior_weights = self.weights_.copy()
        self.posterior_means = self.means_.copy()
        self.posterior_covariances = self.covariances_.copy()
    
    def mle(self):
        """Returns the maximum likelihood estimate of the prices"""
        
        return self.means_.T.dot(self.weights_)
    
    def posterior_mle(self):
        """Returns the maximum likelihood estimate of the posterior prices"""
        
        return self.posterior_means.T.dot(self.posterior_weights)
    
    def std(self):
        """Returns the representative standard deviation of the maximum 
        likelihood estimate of the GMM
        
        Formula adapted from
        https://stats.stackexchange.com/questions/16608/what-is-the-variance-of-the-weighted-mixture-of-two-gaussians
        """

        weighted_cov = self.covariances_.T.dot(self.weights_)
        mean_contrib = (self.means_**2).T.dot(self.weights_) - self.mle()**2
        variance = np.maximum(0,np.diag(weighted_cov)+mean_contrib)
        return np.sqrt(variance)

    def posterior_std(self):
        """Returns the representative standard deviation of the maximum
        likelihood estimate of the posterior GMM after making observations"""

        weighted_cov = self.posterior_covariances.T.dot(
            self.posterior_weights)
        mean_contrib = (self.posterior_means**2).T.dot(self.posterior_weights) \
            - self.posterior_mle()**2
        variance = np.maximum(0,np.diag(weighted_cov)+mean_contrib)
        return np.sqrt(variance)

    def sample(self,n_samples=1,seed=None):
        """Returns n_samples samples from the current distribution. Returns a 
        numpy array of size (n_samples,time_length)"""

        if seed:
            np.random.seed(seed)

        sample = np.zeros((n_samples, self.means_.shape[1]))

        for i in range(n_samples):
            
            # pick which gaussian from weights
            k = np.random.choice(self.means_.shape[0], p=self.weights_)

            # sample from gaussian
            sample[i,:] = np.random.multivariate_normal(self.means_[k,:], self.covariances_[k,:,:])

        return sample

    def posterior_sample(self,n_samples=1,seed=None):
        """Returns n_samples samples from the posterior distribution. Returns a 
        numpy array of size (n_samples,time_length)"""

        if seed:
            np.random.seed(seed)

        sample = np.zeros((n_samples, self.means_.shape[1]))

        for i in range(n_samples):
            
            # pick which gaussian from weights
            k = np.random.choice(self.posterior_means.shape[0], p=self.posterior_weights)

            # sample from gaussian
            sample[i,:] = np.random.multivariate_normal(self.posterior_means[k,:], self.posterior_covariances[k,:,:])

        return sample

    def set_observations(self,observed):
        """Sets the observations variable and updates posterior distributions"""
        
        self.observations = observed
        self.update_posterior()
        return self

    def observe(self,observations):
        """Adds new observations and calculates the posterior 
        distribution."""
        
        observed = np.append(self.observations,observations)
        self.set_observations(observed)
        return self

    def update_posterior(self):
        """Updates posterior distribution over means, weights, and covariances
        based on current observed values stored in self.observations.

        Assumes observations are made in order"""
        
        # do nothing if empty observations
        if len(self.observations)==0:
            return

        # raise error if more observations than possible
        if len(self.observations) > self.means_.shape[1]:
            raise ValueError('Too many observations')
        
        # initialize posteriors appropriately
        xa = self.observations
        seen = len(xa)
        post_weights = self.weights_
        post_cov = np.zeros(self.covariances_.shape)
        post_means = np.zeros(self.means_.shape)

        # add observations to the start of each posterior mean
        post_means[:,:seen] = np.tile(
            xa,(len(self.weights_),1))

        # marginalize distribution
        mua = self.means_[:,:seen]
        mub = self.means_[:,seen:]
        Saa = self.covariances_[:,:seen,:seen]
        Sbb = self.covariances_[:,seen:,seen:]
        Sab = self.covariances_[:,:seen,seen:]
        print(len(self.observations))
        # find each marginal posterior mean/covariance
        for k in range(len(self.weights_)):
            
            # find each non-normalize posterior weight
            post_weights[k] *= stats.multivariate_normal.pdf(
                xa, mean=mua[k], cov=Saa[k], allow_singular=True)

            # update posterior mean and covariance
            post_means[k,seen:] = mub[k] + Sab[k].T.dot(
                np.linalg.inv(Saa[k])).dot(xa-mua[k])

            post_cov[k,seen:,seen:] = Sbb[k] - Sab[k].T.dot(
                np.linalg.inv(Saa[k])).dot(Sab[k])

        self.posterior_weights = post_weights / post_weights.sum()
        self.posterior_means = post_means
        self.posterior_covariances = post_cov
        return self


if __name__ == '__main__':
    import dataAggregator
    gmm = dataAggregator.makeModel(timeRange=[14,34])
    eGMM = energyGMM(gmm, time_range=[14,34])
    print(eGMM.mle())
    print(eGMM.std())
    print(eGMM.sample())
    print(eGMM.sample(3))
    eGMM.observe(1.05)

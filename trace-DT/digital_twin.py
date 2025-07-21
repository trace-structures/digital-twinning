import emcee
import time
import pandas as pd
import numpy as np
from multiprocessing import Pool
import utils

class DigitalTwin:
    def __init__(self, model, E):
        self.model = model
        self.E = E
        self.Q = model.Q
        
        
    def get_logprob(self, q):
        if self.Q_ == 'default':
            logprob = self.loglikelihood(q, self.y_m) + self.logprior(q)
            return logprob
        else:
            logprob = self.loglikelihood(q, self.y_m) + self.Q_.logpdf(q)
            return logprob 
        
    def update(self, y_m, nwalkers=150, nburn=100, niter=350, Q_='default', plot_samples=False, name_pairs=None):
        self.Q_ = Q_
        if name_pairs is not None:
            H = self.model.create_mx_from_tuple(name_pairs, list(y_m.keys()))
            self.q_indices, y_indices = self.get_indices_from_mx(H)
            y_m = y_m.to_numpy().reshape(-1, 1)[y_indices]
            self.E = self.E.diminished_paramset(y_indices)
            # self.Q = self.Q.diminished_paramset(self.q_indices)
        else:
            self.q_indices = None
            y_m = y_m.to_numpy()
        self.y_m = y_m
        num_param = self.Q.num_params()
        
        if Q_ == 'default':
            #logprob = lambda q: self.loglikelihood(q, y_m) + self.logprior(q)
            p0 = self.Q.sample(nwalkers)
        else:
            #logprob = lambda q: self.loglikelihood(q, y_m) + Q_.logpdf(q)
            p0 = Q_.sample(nwalkers)
        self.p0 = p0

        
        #with Pool(4) as pool:
        print('MCMC creating')
        sampler = emcee.EnsembleSampler(nwalkers, num_param, self.get_logprob)#, pool=pool)
        start_time = time.time()

        print('Burning period')
        state = sampler.run_mcmc(p0, nburn, progress = True)
        sampler.reset()

        print('MCMC running')
        sampler.run_mcmc(state, niter, progress = True)
    
        print("--- %s seconds ---" % (time.time() - start_time))
        self.sampler = sampler
        #pool.close()
            
        if plot_samples:
            # TODO sns pairplot
            pass
        
    def likelihood(self, q, y_m):
        # TODO inverse_transform to predicted data
        #q = self.model.get_scaled_q(q.reshape(1,-1))
        q_df = pd.DataFrame(q.reshape(1,-1), columns=self.Q.param_names())
        d = y_m - self.model.predict(q_df)
        d = d.transpose()
        p = self.E.pdf(d)
        return p
    
    def loglikelihood(self, q, y_m):
        #q = self.model.get_scaled_q(q)
        q_df = pd.DataFrame(q.reshape(1,-1), columns=self.Q.param_names())
        if self.q_indices is None:
            d = y_m - self.model.predict(q_df)
            d = d.transpose()
        else:
            d = y_m - self.model.predict(q_df).reshape(-1, 1)[self.q_indices]
        logp = self.E.logpdf(d)
        return logp
    
    def prior(self, q):
        #q = self.model.get_scaled_q(q)
        pr = self.Q.pdf(q.reshape(1,-1))
        return pr
    
    def logprior(self, q):
        #q = self.model.get_scaled_q(q)
        logpr = self.Q.logpdf(q.reshape(-1,1))
        return logpr
    
    def get_mean_and_var_of_posterior(self):
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        means = np.mean(post_samples, axis=0)
        variances = np.var(post_samples, axis=0)
        means_df = pd.DataFrame(means.reshape(1,-1), columns=self.Q.param_names())
        variances_df = pd.DataFrame(variances.reshape(1,-1), columns=self.Q.param_names())
        return means_df, variances_df
    
    def get_MAP(self): # maximum a posterior estimate
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        
        map_estimate = np.zeros((1, post_samples.shape[1]))
        for p in range(post_samples.shape[1]):
            p_estimate = utils.estimate_maxima(post_samples[:,p])
            map_estimate[0,p] = p_estimate

        map_df = pd.DataFrame(map_estimate, columns=self.Q.param_names())
        return map_df
    
    def get_indices_from_mx(self, H):
        q_indices = []
        y_indices = []
        for i in range(H.shape[1]):
            if 1 in H[:, i]:
                q_indices.append(np.where(H[:, i]==1)[0][0])

        for i in range(H.shape[0]):
            if 1 in H[i, :]:
                y_indices.append(np.where(H[i, :]==1)[0][0])

        return q_indices, y_indices
                
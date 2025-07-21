from distributions import *
from simparameter import SimParameter
from simparameter_set import SimParamSet
from surrogate_model import SurrogateModel
import pandas as pd
import emcee
import time
import utils

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


class JointManager:
    def __init__(self, models, joint_parameters):
        self.models = models
        self.joint_parameters = joint_parameters
        self.Q, self.indices, self.scale, self.shift = self.get_joint_paramset_and_indices(models, joint_parameters)

    def generate_joint_stdrn_simparamset(self, sigmas):
        joint_sigma = np.zeros((1, 0))
        for i in range(len(sigmas)):
            building_sigma = sigmas[i].to_numpy()
            joint_sigma = np.concatenate((joint_sigma, building_sigma), axis=1)
        joint_sigma = joint_sigma.reshape(-1, 1)
        E_joint = utils.generate_stdrn_simparamset(joint_sigma)
        return E_joint
    
    def choose_distribution(self, buildings, joint_names, mode='auto'):
        if mode == 'first_dist':
            for i in range(len(joint_names)):
                if joint_names[i] != '-':
                    break
            distribution = buildings[i].Q.params[joint_names[i]]
            return distribution
        
        elif mode == 'uniform':
            distribution = UniformDistribution(0, 1)
            return distribution
        
        elif mode == 'normal':
            distribution = NormalDistribution(0, 1)
            return distribution
        
        elif mode == 'auto':
            for i in range(len(joint_names)):
                if joint_names[i] != '-':
                    break
            if buildings[i].Q.params[joint_names[i]].get_type() == 'Uniform':
                distribution = UniformDistribution(0, 1)
                return distribution
            elif buildings[i].Q.params[joint_names[i]].get_type() == 'Normal':
                distribution = NormalDistribution(0, 1)
                return distribution                



    
    def get_joint_paramset_and_indices(self, buildings, joint_parameters):
        building_parameters = []
        for b in range(len(buildings)):
            building_parameters.append(buildings[b].Q.param_names())
        joint_names = list(joint_parameters.keys())
        for i in range(len(joint_parameters)):
            parameter_list = joint_parameters[joint_names[i]]
            assert len(parameter_list) == len(building_parameters), "The jointparameters should contain every building. Put a '-' when you don't want to use the building for parameter join"
            for j in range(len(building_parameters)):
                if parameter_list[j] != '-':
                    assert parameter_list[j] in building_parameters[j], "Parameter '{}' is not in parameter list {}".format(parameter_list[j], building_parameters[j])

        Q_joint = SimParamSet()

        indices = []
        scale = []
        shift = []
        for i in range(len(building_parameters)):
            indices.append([None]*len(building_parameters[i]))
            scale.append([1]*len(building_parameters[i]))
            shift.append([0]*len(building_parameters[i]))
        
        for i in range(len(joint_parameters)):
            distribution = self.choose_distribution(buildings, joint_parameters[joint_names[i]], mode='auto')
            JointParameter = SimParameter(joint_names[i], distribution)
            Q_joint.add(JointParameter)
            scale0 = distribution.get_bounds()[1] - distribution.get_bounds()[0]
            shift0 = distribution.get_bounds()[0]
            for j in range(len(building_parameters)):
                param_name = joint_parameters[joint_names[i]][j]
                if param_name != '-':
                    idx = buildings[j].Q.param_names().index(param_name)
                    indices[j][idx] = i
                    
                    lower_bound = buildings[j].Q.get_bounds()[idx][0]
                    higher_bound = buildings[j].Q.get_bounds()[idx][1]
                    scale[j][idx] = (higher_bound - lower_bound) / scale0
                    shift[j][idx] = lower_bound - scale0*shift0
                    building_parameters[j].remove(param_name)

        for i in range(len(building_parameters)):
            for j in range(len(building_parameters[i])):
                Parameter = SimParameter(building_parameters[i][j], buildings[i].Q.params[building_parameters[i][j]])    
                Q_joint.add(Parameter)

        for i in range(len(building_parameters)):
            for j in range(len(building_parameters[i])):
                parameter = building_parameters[i][j]
                idx = Q_joint.param_names().index(parameter)
                original_index = buildings[i].Q.param_names().index(parameter)
                indices[i][original_index]  = idx   
        
        return Q_joint, indices, scale, shift
    
    def get_logprob(self, q):
        if self.Q_ == 'default':
            logprob = self.loglikelihood(q, self.y_m) + self.logprior(q)
            return logprob
        else:
            logprob = self.loglikelihood(q, self.y_m) + self.Q_.logpdf(q)
            return logprob
        
    def logprior(self, q):
        #q = self.model.get_scaled_q(q)
        logpr = self.Q.logpdf(q.reshape(-1,1))
        return logpr
    
    def loglikelihood(self, q, y_m):
        #q = self.model.get_scaled_q(q)
        q_df = pd.DataFrame(q.reshape(1,-1), columns=self.Q.param_names())
        predictions = np.zeros((1, 0))
        for i in range(len(self.models)):
            q_model = q[self.indices[i]] * self.scale[i] + self.shift[i]
            q_df = pd.DataFrame(q_model.reshape(1,-1), columns=self.models[i].Q.param_names())
            y_model = self.models[i].predict(q_df)
            predictions = np.concatenate((predictions, y_model), axis=1)
        d = y_m - predictions
        d = d.transpose()
        logp = self.E.logpdf(d)
        return logp
    
    def update(self, y_list, sigmas, nwalkers=150, nburn=100, niter=350, Q_='default', plot_samples=False):
        self.Q_ = Q_
        y_m = np.zeros((1, 0))
        for i in range(len(y_list)):
            y_building = y_list[i].to_numpy()
            y_m = np.concatenate((y_m, y_building), axis=1)
        # y_m = y_m.to_numpy()
        self.y_m = y_m
        num_param = self.Q.num_params()
        self.E = self.generate_joint_stdrn_simparamset(sigmas)


        if Q_ == 'default':
            #logprob = lambda q: self.loglikelihood(q, y_m) + self.logprior(q)
            p0 = self.Q.sample(nwalkers)
        else:
            #logprob = lambda q: self.loglikelihood(q, y_m) + Q_.logpdf(q)
            p0 = Q_.sample(nwalkers)
        self.p0 = p0

        
        # with Pool(4) as pool:
        print('MCMC creating')
        sampler = emcee.EnsembleSampler(nwalkers, num_param, self.get_logprob) #pool=pool
        start_time = time.time()

        print('Burning period')
        state = sampler.run_mcmc(p0, nburn, progress = True)
        sampler.reset()

        print('MCMC running')
        sampler.run_mcmc(state, niter, progress = True)
    
        print("--- %s seconds ---" % (time.time() - start_time))
        self.sampler = sampler
            # pool.close()
            
    def get_MAP(self, m): # maximum a posterior estimate
        sampler = self.sampler
        post_samples = sampler.get_chain(flat=True)
        log_probs = sampler.get_log_prob(flat=True)

        # Find the index of the highest log probability
        max_idx = np.argmax(log_probs)
        
        # Extract the corresponding sample
        map_estimate = post_samples[max_idx]
        
        # Convert to DataFrame
        map_df = pd.DataFrame(map_estimate.reshape(1, -1), columns=self.Q.param_names())
        return map_df

def main():
    ######################################
    #           Palisaden model:         #
    ######################################

    print('Training Palisaden model')

    x_df = pd.read_csv('C:/Users/Bence/new_sglib_ttb/sglib_ttb/demo/data/PS_data/x_df.csv')
    y_df = pd.read_csv('C:/Users/Bence/new_sglib_ttb/sglib_ttb/demo/data/PS_data/y_df.csv')

    param_names = ['COEF_VERT', 'COEF_G', 'FACADE', 'FOUNDATION', 'WEIGHT']
    a = [0.5, 1, 1, 7, 0.8]
    b = [1, 3, 1.5, 10, 1.2] #1. changed to 1 from 1.5

    P1 = SimParameter('COEF_VERT', UniformDistribution(0.5, 1))
    P2 = SimParameter('COEF_G', UniformDistribution(1, 3))
    P3 = SimParameter('FACADE', UniformDistribution(1, 1.5))
    P4 = SimParameter('FOUNDATION', UniformDistribution(7, 10))
    P5 = SimParameter('WEIGHT', UniformDistribution(0.8, 1.2))

    PQ = SimParamSet()
    PQ.add(P1)
    PQ.add(P2)
    PQ.add(P3)
    PQ.add(P4)
    PQ.add(P5)


    config = {
                'init_config' : {
                'p' : 5
                }, 
                'train_config' : {
                    'k_fold': 9
                    }
            }

    split_config = {
            'train_test_ratio': 0.2, 
            'random_seed': 1997,
            'split_type': 'no_shuffle'
            }

    Palisaden_model = SurrogateModel(PQ, PQ.param_names(), 'gPCE', **config['init_config'])  
    Palisaden_model.train_test_split(x_df, y_df, **split_config)
    Palisaden_model.train(Palisaden_model.X_train, Palisaden_model.y_train, **config['train_config'])


    ########################################
    #       Yoker model                    #
    ########################################

    print('Training Yoker model')

    param_names = ['e1', 'e2', 'e3', 'g1', 'g2', 'q']
    a = [6, 10, 6, 400, 200, 5]
    b = [12, 13, 12, 750, 500, 100]

    YQ = SimParamSet()

    P1 = SimParameter('e1', UniformDistribution(6, 12))
    P2 = SimParameter('e2', UniformDistribution(10, 13))
    P3 = SimParameter('e3', UniformDistribution(6, 12))
    P4 = SimParameter('g1', UniformDistribution(400, 750))
    P5 = SimParameter('g2', UniformDistribution(200, 500))
    P6 = SimParameter('q', UniformDistribution(5, 100))

    YQ.add(P1)
    YQ.add(P2)
    YQ.add(P3)
    YQ.add(P4)
    YQ.add(P5)
    YQ.add(P6)

    x_df = pd.read_csv('C:/Users/Bence/new_sglib_ttb/sglib_ttb/demo/data/YB_data/x_df.csv')
    y_df = pd.read_csv('C:/Users/Bence/new_sglib_ttb/sglib_ttb/demo/data/YB_data/y_df.csv')

    config = {
                'init_config' : {
                'p' : 5
                }, 
                'train_config' : {
                    'k_fold': 9
                    }
            }

    split_config = {
            'train_test_ratio': 0.2, 
            'random_seed': 1997,
            'split_type': 'no_shuffle'
            }

    Yoker_model = SurrogateModel(YQ, YQ.param_names(), 'gPCE', **config['init_config'])  
    Yoker_model.train_test_split(x_df, y_df, **split_config)
    Yoker_model.train(Yoker_model.X_train, Yoker_model.y_train, **config['train_config'])


    ###########################################
    #           Read joint parameters         #
    ###########################################

    joint_parameters = {'joint_e1_COEF_VERT': ['e1', 'COEF_VERT']}


    ##########################################
    #           Create JointManager instance #
    ##########################################

    jointManager = JointManager([Yoker_model, Palisaden_model], joint_parameters)
    print('JointManager created')

    ###########################################
    #       Get data for update               #
    ###########################################

    sigma_palisaden = pd.read_csv('C:/Users/Bence/new_sglib_ttb/sglib_ttb/demo/data/PS_data/sigma.csv')
    sigma_yoker = pd.read_csv('C:/Users/Bence/new_sglib_ttb/sglib_ttb/demo/data/YB_data/sigma.csv')

    z_palisaden = pd.read_csv('C:/Users/Bence/new_sglib_ttb/sglib_ttb/demo/data/PS_data/z_m_df.csv')
    z_yoker = pd.read_csv('C:/Users/Bence/new_sglib_ttb/sglib_ttb/demo/data/YB_data/z_m_df.csv')

    ##############################################
    #           Multibuilding Update             #
    ##############################################

    print('Joint update started')
    jointManager.update([z_palisaden, z_yoker], [sigma_palisaden, sigma_yoker])

    ###############################################
    #           Save results                      #
    ###############################################

    

    ###############################################
    #            Plot results                     #
    ###############################################

    utils.plot_multibuilding_MCMC(jointManager)
    # print(figures)

if __name__ == "__main__":
    main()
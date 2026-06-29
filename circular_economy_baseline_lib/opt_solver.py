import numpy as np
import xpress as xp
from surrogate_models import SurrogateModel
import tensorflow as tf

class XpressSolver(object):
    def __init__(self, num_agents: int, num_commodities: int):
        self.num_agents = num_agents
        self.num_commodities = num_commodities

        self.surrogate_model = SurrogateModel()

        self.agent_surrogate_final_output_ids = {0: [2,6,0], 1: [11,5], 2: [2,8,3,0]}

        self.agent_surrogate_model_output_ids = {0: [0,1,3], 1: [0,1], 2: [0,1,3,4]}

        self.agent_surrogate_input_ids = {0: [5,7,2,0,1], 1: [3,4,10,9], 2: [0,2]}

    def solve_transformation(self, agent_id, params):
        m = xp.problem()
        # Parameters
        num_commodities = self.num_commodities 
        delta = params['delta']
        e = params['prices']
        ew = params['waste_prices']
        p = params['uc']
        pr = params['tx']
        bar_inv = params['balanced_inv']
        bar_winv = params['balanced_winv']

        # Variables
        uec = [xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)]

        #only create variables for relevant commodities of current agent
        utx_ = [xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)]

        if agent_id != 2:
            utx = [utx_[i] for i in self.agent_surrogate_input_ids[agent_id]]
        else:
            utx = [1./9.*sum([utx_[i] for i in self.agent_surrogate_input_ids[agent_id]])]


        m.addVariable(uec, utx_)

        layer_outputs = utx
        bn_index = 0

        weights, biases, bn_params, layer_type_list = self.surrogate_model.get_agent_model_params(agent_id)
        model = self.surrogate_model.get_agent_model(agent_id)

        idx1 = 0
        idx2 = 0
        for i in range(len(layer_type_list)):
            if layer_type_list[i]:
                gamma, beta, mean, variance = bn_params[idx1]
                next_layer = []
                for j in range(len(gamma)):
                    bn_output = gamma[j] * (layer_outputs[j] - mean[j]) / np.sqrt(variance[j] + 1e-5) + beta[j]
                    next_layer.append(bn_output)
                layer_outputs = next_layer
                idx1 += 1
            else:
                next_layer = []
                w, b = weights[idx2], biases[idx2]
                for j in range(len(b)):
                    linear_combination = sum(w[k, j] * layer_outputs[k] for k in range(len(layer_outputs))) + b[j]
                    next_layer.append(linear_combination)
                layer_outputs = next_layer
                idx2 += 1




        output_vars = []

        #m.addVariable(output_vars)

        for i in range(len(self.agent_surrogate_model_output_ids[agent_id])):
            output_vars.append(layer_outputs[self.agent_surrogate_model_output_ids[agent_id][i]])



        # Helper
        # TODO-Use the real surrogate model implementation
        trans_u, trans_w = [], []
        for i in range(num_commodities):
            trans_u.append(0.)
            trans_w.append(0.)

        for i in range(len(self.agent_surrogate_model_output_ids[agent_id])):
            trans_u[self.agent_surrogate_final_output_ids[agent_id][i]] = output_vars[i]

        f_inv = [] 
        f_winv = [] 
        for i in range(num_commodities):
            f_inv.append(bar_inv[i]-uec[i]-utx_[i]+trans_u[i])
            f_winv.append((1-delta) * (bar_winv[i]+trans_w[i]))

        # Constraints
        for i in range(num_commodities):
            m.addConstraint(uec[i] + utx_[i] <= bar_inv[i]) # TODO: adjust the 0.8

        # Objective
        m.setObjective(xp.Sum(p[c]*uec[c] - pr[c]*utx_[c] + delta*e[c]*f_inv[c] +\
                delta*ew[c]*f_winv[c] for c in range(num_commodities)), sense=xp.maximize)
        
        # Optimize
        m.solve()
        status = m.getProbStatus()

        eco_utility = np.zeros(num_commodities)
        trans_quan = np.zeros(num_commodities)

        for i in range(num_commodities):
            eco_utility[i]=m.getSolution(uec[i])
            trans_quan[i]=m.getSolution(utx_[i])


        return m.getObjVal(), eco_utility, trans_quan

    def solve_buyer(self, params, agent):

        m = xp.problem()
        # Parameters
        num_commodities = self.num_commodities 
        num_other_ind = self.num_agents-1
        inv = params['inv']
        invw = params['waste_inv']
        max_inv = params['max_inv']
        max_invw = params['max_waste_inv']
        beta = params['beta']
        e = params['prices']
        ew = params['waste_prices']
        p = params['spot_price']

        #import pdb; pdb.set_trace()


        # Variables
        q = [[xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)] for _ in range(num_other_ind)]
        qw = [[xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)] for _ in range(num_other_ind)]
        qs = [xp.var(lb=0, vartype=xp.continuous) for _ in range(num_commodities)]

        m.addVariable(q, qw, qs)
        # Helper
        qsum = [xp.Sum(q[n][c] for n in range(num_other_ind)) for c in range(num_commodities)]
        qwsum = [xp.Sum(qw[n][c] for n in range(num_other_ind)) for c in range(num_commodities)]

        # Constraints
        for c in range(num_commodities):
            m.addConstraint(inv[c]+0.1*qsum[c]+qs[c]<=max_inv[c])
            m.addConstraint(inv[c]+0.1*qsum[c]+qs[c]>=0.2*max_inv[c])
            m.addConstraint(invw[c]+0.1*qwsum[c]<=max_invw[c])
            m.addConstraint(invw[c]+0.1*qwsum[c]>=0.2*max_invw[c])
            m.addConstraint(qs[c] >= 3.*qsum[c])

        e_idxes = []
        for i in range(num_other_ind):
            if i < agent:
                e_idxes.append(i)
            else:
                e_idxes.append(i+1)

        # Objective
        m.setObjective(xp.Sum(-0.1*q[n][c]*e[e_idxes[n]][c]-0.1*qw[n][c]*ew[e_idxes[n]][c] for c in range(num_commodities)\
                for n in range(num_other_ind)) + xp.Sum(-qs[c]*p[c] for c in range(num_commodities))\
                + beta*xp.Sum(e[agent][c]*(inv[c]+0.1*qsum[c]+qs[c])+ew[agent][c]*(invw[c]+0.1*qwsum[c])\
                for c in range(num_commodities)), sense=xp.maximize)
        
        # Optimize
        m.solve()
        status = m.getProbStatus()
        explanation = m.getProbStatusString()

        # Return
        quantity  = np.zeros((num_other_ind, num_commodities))
        quantityw = np.zeros((num_other_ind, num_commodities))
        quantitys = np.zeros(num_commodities)

        for c in range(num_commodities):
            for n in range(num_other_ind):
                quantity[n][c]=m.getSolution(q[n][c])
                quantityw[n][c]=m.getSolution(qw[n][c])
            quantitys[c]=m.getSolution(qs[c])

        #import pdb; pdb.set_trace()
        return m.getObjVal(), quantity, quantityw, quantitys

    def solve_seller(self, params):
        m = xp.problem()
        # Parameters
        num_commodities = self.num_commodities 
        avg_spot_price = params['avg_spot_price']
        inv = params['inv']
        waste_inv = params['waste_inv']
        alpha = params['alpha']
        phi = params['phi']

        # Variables
        e = [xp.var(lb=0, ub=avg_spot_price[i], vartype=xp.continuous) for i in range(num_commodities)]
        ew = [xp.var(lb=0, ub=phi*avg_spot_price[i], vartype=xp.continuous) for i in range(num_commodities)]
        carry_inv = [xp.var(lb=0, ub=inv[i], vartype=xp.continuous) for i in range(num_commodities)]
        carry_waste = [xp.var(lb=0, ub=waste_inv[i], vartype=xp.continuous) for i in range(num_commodities)]

        m.addVariable(e, ew, carry_inv, carry_waste)
        
        for i in range(num_commodities):
            m.addConstraint(ew[i] <= 0.5*e[i])

        # Objective
        m.setObjective(xp.Sum([e[i]*(inv[i]-carry_inv[i])+ew[i]*(waste_inv[i]-carry_waste[i])+\
                alpha*(e[i]*carry_inv[i]+ew[i]*carry_waste[i]) for i in range(num_commodities)]),\
                sense=xp.maximize)

        #Optimize
        m.solve()
        # status = m.getProbStatus()
        # explanation = m.getProbStatusString()
        #if (status != OptimizationStatus.OPTIMAL and status != OptimizationStatus.FEASIBLE):
        #    return "Infeasible", m.objective_value, None, None, None, None 

        price = np.zeros(num_commodities)
        waste_price = np.zeros(num_commodities)
        inv = np.zeros(num_commodities)
        waste_inv = np.zeros(num_commodities)
        for i in range(num_commodities):
            price[i]=m.getSolution(e[i])
            waste_price[i]=m.getSolution(ew[i])
            inv[i]=m.getSolution(carry_inv[i])
            waste_inv[i]=m.getSolution(carry_waste[i])


        return m.getObjVal(), price, waste_price, inv, waste_inv

    def step1(self, state):
        obj = np.zeros(self.num_agents)
        price = np.zeros((self.num_agents, self.num_commodities)) 
        waste_price = np.zeros((self.num_agents, self.num_commodities)) 
        inv = np.zeros((self.num_agents, self.num_commodities))
        waste_inv = np.zeros((self.num_agents, self.num_commodities))
        for n in range(self.num_agents):
            o, e, ew, i, iw = self.solve_seller(state[n])
            price[n,:] = e
            waste_price[n,:] = ew
            inv[n,:] = i
            waste_inv[n,:] = iw
            obj[n] = o
        
        actions = np.concatenate([price, waste_price], axis=1)

        return obj, actions
    
    def step2(self, state):
        obj = np.zeros(self.num_agents)
        quantity = np.zeros((self.num_agents, self.num_agents-1, self.num_commodities))
        quantityw = np.zeros((self.num_agents, self.num_agents-1, self.num_commodities))
        quantitys = np.zeros((self.num_agents, self.num_commodities))
        for n in range(self.num_agents):
            o, q, qw, qs = self.solve_buyer(state[n], n)
            quantity[n, :] = q
            quantityw[n, :] = qw
            quantitys[n, :] = qs
            obj[n] = o
        
        action = np.concatenate([quantity.reshape((self.num_agents, -1)), quantityw.reshape((self.num_agents, -1)), quantitys], axis=1)

        return obj, action
    
    def step3(self, state):
        obj = np.zeros(self.num_agents)
        eco_utility = np.zeros((self.num_agents, self.num_commodities)) 
        trans_quan = np.zeros((self.num_agents, self.num_commodities))
        
        for n in range(self.num_agents):
            o, uec, utx = self.solve_transformation(n, state[n])
            eco_utility[n, :] = uec
            trans_quan[n,:] = utx
            obj[n] = o
        
        action = np.concatenate([trans_quan, eco_utility], axis=1)

        return obj, action


import numpy as np
from evCosts import *
from evTransitions import *


# electric vehicle mdp class
class evMDP:
    
    # class constructor
    def __init__(self, num_cars, simTime=[16,32], dt=0.25, reward={}, transitions={}):
        
        ### General Parameters
        
        self.n = num_cars
        # self.cl = charge_levels # lets assume continuous charge (0,1)
        self.times = simTime
        self.dt = dt
        
        ### Cost function Modeling
        
        # Default Parameters
        self.cost_function = TimeOfUse
        self.cost_model = None
        self.cost_exp = 1.
        self.terminal_weighting = 100.
        
        # Overrides
        if 'cost_function' in reward.keys():
            self.cost_function = reward['cost_function']
        if 'cost_model' in reward.keys():
            self.cost_model = reward['cost_model']
        if 'cost_exp' in reward.keys():
            self.cost_exp = reward['cost_exp']             
        if 'terminal_weighting' in reward.keys():
            self.terminal_weighting = reward['terminal_weighting']
        
        ## Car Transition Modeling
        
        # Default Parameters
        self.transition_model = Residential
        self.model_parameters = {'base_seed': 1995}
        
        # Overrides
        if 'transition_model' in transitions.keys():
            self.transition_model = transitions['transition_model']
            try:
                self.model_parameters = transitions['model_parameters']
            except:
                print('Missing Car Transition Model Parameters')
                raise
        
        ### Miscellaneous Parameters
        
        # assume constant capacity batteries
        self.batteries = 75 # [kWh] - 2019 Tesla Model 3

    # propagate a state forward given actions with the parameters of the mdp
    def propagateState(self, state, actions, requests=None):
        
        # default requests of one-fifth charge level / hour
        if requests==None:
            requests = 0.2*self.dt*np.ones(self.charges.shape)
        
        assert (requests.shape == self.charges.shape), "Incorrect request shape"
        assert (actions.shape == self.charges.shape), "Incorrect actions shape"
        
        # assert that you can only charge cars that are present
        # assert (actions <= 1+state.charges)
        # determine charge changes
        charge_change = requests*actions
        
        # determine next charge
        new_charges = state.charges + charge_change
        
        # clip at 1
        new_charges = np.minimum(new_charges, 1.)
        
        # expected next state
        sp_expected = evState(state.t + self.dt, new_charges)
        
        # but cars may leave or arrive
        sp = self.transition_model(sp_expected, self.model_parameters)
        
        # get reward
        reward = self.getReward(state,actions,sp)
        
        r = reward['totalCost']
        return sp, r
    
    # obtain reward for taking actions s,a and ending in sp with the params of the mdp
    def getReward(self, s, a, sp):     
        reward = {}
        
        # unit cost of electricity found with call to model defined in mdp
        electricUnitCost = self.cost_function(s.t, self.cost_model)
        
        # total consumption found by multiplying battery storage in each car by charge differences
        totalConsumption = sum(self.batteries*(sp.charges-s.charges)*a)
        
        # total electric cost
        reward['electricCost'] = -electricUnitCost*totalConsumption
        
        # total terminal state cost (associated with cars leaving)
        reward['terminalCost'] = getTerminalCost(s,a,sp)
        
        reward['totalCost'] = reward['electricCost'] + self.terminal_weighting*reward['terminalCost']
        
        return reward
    
# electric vehicle states. Includes current time, and charge levels
# of all cars in state
class evState:
    
    # class constructor charge of -1 means no car present
    def __init__(self,time,charges):
        self.t = time
        self.charges=charges
        
    # overload == operator
    def __eq__(self,state):
        return self.t == state.t and self.charges == state.charges
        

# might not be necessary
class evAction:
    
    def __init__(self,actions):
        self.action = actions

    
def main():
    testMDP = evMDP(10,reward = {'cost_function': TimeOfUse, 'cost_model':"PG&E Summer"})
    
    
if __name__ == '__main__':
    main()
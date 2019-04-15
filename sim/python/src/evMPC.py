import numpy as np
import cvxpy as cp
from energyGMM import *
from evChargeState import *

class evMPC(object):
    """
    Run stochastic MPC for an overnight charging simulation.

    Includes functions to initialize a charge state array for multiple cars at a 
    given timestep, to update charge state given some actions, and to form
    cvxpy-style costs and constraints to pass to a solver

    Attributes:
        state: current state of charge
        demand_model: energyGMM-class base demand model prior/posterior

        end_times: time of departure for each car
        rise: per car electric rise
        eta: dual weighting parameter (applied to electric costs)

        past_times: (# timesteps, )-sized numpy array of already visited
            simulation times
        actions_taken: (# cars, # timesteps)-sized numpy array of actions
            taken so far
            


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

    def __init__(self, demand_model, cars, t=18, charge=0.4, 
        charge_rate=1./12, end_times=31, rise=0.1, eta=0.05):
        """Initializes MPC class.
        
        Args:
            
            t: initial time for simulation
            cars: number of cars in the simulation
            charge: initial charge state - can be single value or array of 
                length cars
            charge_rate: 

        Raises:
            ValueError if charge or chargeRate do not have length cars
        """

        self.charge_state = evChargeState(t, cars, charge, charge_rate)
        
        # add check for energyGMM demand model
        self.demand_model = demand_model
        
        if isinstance(end_times,(int,float)):
            self.end_times = np.array([end_times]*cars)
        elif isinstance(charge, np.ndarray):
            self.end_times = end_times
        else:
            raise ValueError('Incorrect end_times type')
        
        self.rise = rise 
        self.eta = eta 
        
        self.past_times = np.empty((0,), float)
        self.actions_taken = np.empty((cars,0), int)
        
        self.times_remaining = np.arange(t, self.end_times.max())
        self.actions_remaining = np.zeros((cars,len(self.times_remaining)))

    def step(self, observations, dt, relaxation=False, solver=cp.GUROBI):
        """Perform MPC up through the next observed base demand
        
        Args:
            observations: numpy array or float of new hourly observations
            dt: time step over which to perform actions
            relaxation: whether or not to solve the relaxed problem
                (default: False)
            solver: cvxpy-compatible solver to solve optimization problem
                (default: cp.GUROBI)

        Raises:
            ValueError 1/dt is not (approximately) integral 
        """

        # use SCS if we wish to do relaxation
        if relaxation:
            solver = cp.SCS
        
        # set keyword argument dictionaries (to be unpacked in solver call)
        if solver == cp.SCS:
            solver_kwargs = {'max_iters': 20000} 
        elif solver == cp.GUROBI:
            solver_kwargs = {'TimeLimit': 20.} 

        # update demand model with observations
        self.demand_model = self.demand_model.observe(observations)

        # generate D from updated demand model
        D = np.vstack(
            (self.demand_model.time_range, self.demand_model.posterior_mle()))

        # generate optimization problem
        var, cost, constraints = self.charge_state.optimization_problem(
            D, self.end_times, self.rise, self.eta, dt, relaxation=relaxation)
        prob = cp.Problem(cp.Minimize(cost), constraints)

        # solve optimization problem
        pstar = prob.solve(solver=solver, verbose=True, **solver_kwargs)
        policy = var.value

        # decide actions to take now vs. store for later as reference
        acts = round(1./dt)
        if abs(1./dt - acts) >= 1e-2:
            raise ValueError('Use dt that fits evenly into 1')
        actions = policy[:,:acts]            
        self.actions_remaining = policy[:,acts:]
        self.actions_taken = np.hstack((self.actions_taken,actions))

        # adjust times visited and times remaining
        times = np.arange(self.charge_state.t, self.end_times.max(), dt)
        self.past_times = np.hstack((self.past_times,times[:acts]))
        self.times_remaining = times[acts:] 

        # update charge state with actions
        self.charge_state.update(actions, dt)
        
    def simulate(self, base_demand, dt):
        """Simulate MPC through end times"""
        
        for observation in base_demand:
            self.step(observation, dt)

    
    def total_demand(self):
        """Return both past and future estimated total demand
        
        Returns:
            past: (2, # past timesteps)-sized array with the first row w/
                past timesteps, and the second row w/ total past demand
            future: (2, # future timesteps)-sized array with the first row
                w/ future timesteps, and the second row w/ total future demand
        """

        past_base_demand = np.interp(self.past_times,
            self.demand_model.time_range, self.demand_model.posterior_mle())
        future_base_demand = np.interp(self.times_remaining,
            self.demand_model.time_range, self.demand_model.posterior_mle())

        past_total_demand = past_base_demand+self.rise*self.actions_taken.sum(
            axis=0)
        future_total_demand = future_base_demand+self.rise \
            * self.actions_remaining.sum(axis=0)

        past = np.vstack((self.past_times, past_total_demand))
        future = np.vstack((self.times_remaining,future_total_demand))
        return past, future

    def charging_ratio(self):
        """Return both past and future estimated ratio of cars charging
        
        Returns:
            past: (2, # past timesteps)-sized array with the first row w/
                past timesteps, and the second row w/ ratio of cars charging
            future: (2, # future timesteps)-sized array with the first row
                w/ future timesteps, and the second row w/ ratio of cars charging
        """

        past_ratio = self.actions_taken.sum(
            axis=0) / self.actions_taken.shape[0]
        future_ratio = self.actions_remaining.sum(
            axis=0) / self.actions_remaining.shape[0]
        
        past = np.vstack((self.past_times, past_ratio))
        future = np.vstack((self.times_remaining,future_ratio))
        return past, future

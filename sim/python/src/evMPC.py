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
        state - current state of charge
        demand_model - energyGMM-class base demand model prior/posterior

        end_times - time of departure for each car
        rise - per car electric rise
        eta - dual weighting parameter (applied to electric costs)

        past_times - (# timesteps, )-sized numpy array of already visited
            simulation times
        actions_taken - (# cars, # timesteps)-sized numpy array of actions
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
            
            t - initial time for simulation
            cars - number of cars in the simulation
            charge - initial charge state - can be single value or array of 
                length cars
            charge_rate - 

        Raises:
            ValueError if charge or chargeRate do not have length cars
        """

        self.charge_state = evChargeState(t, cars, charge, charge_rate)
        
        # add check for energyGMM demand model
        self.demand_model = demand_model
        
        self.end_times = end_times
        self.rise = rise 
        self.eta = eta 
        
        self.past_times = np.empty((0,), float)
        self.actions_taken = np.empty((cars,0))

    def step(self, observations, dt, relaxation=False, solver=cp.GUROBI):
        """Perform MPC up through the next observed base demand
        
        Args:
            
            observations - numpy array or float of new hourly observations
            dt - time step over which to perform actions
            relaxation - whether or not to solve the relaxed problem
                (default: False)
            solver - cvxpy-compatible solver to solve optimization problem
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
            solver_kwargs = {'TimeLimit': 10.} 

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
        pstar = prob.solve(solver=solver, **solver_kwargs)
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
        
    def simulate(self):
        """Simulate MPC through end times"""
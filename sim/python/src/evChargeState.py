import numpy as np
import cvxpy as cp

class evChargeState(object):
    """
    Saves relevant information about state of charge for a set of cars.

    Includes functions to initialize a charge state array for multiple cars at a 
    given timestep, to update charge state given some actions, and to form
    cvxpy-style costs and constraints to pass to a solver

    Attributes:
        cars
        charge
        time
        charge_rates


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

    def __init__(self, t=18, cars=10, charge=0.4, charge_rate=1./12):
        """Initializes EV charge state class.
        
        Args:
            t - initial time for simulation
            cars - number of cars in the simulation
            charge - initial charge state - can be single value or array of 
                length cars
            charge_rate - rate at which each car charges 

        Raises:
            ValueError if charge or charge_rate do not have length cars
        """

        self.t = t
        self.cars = cars

        if isinstance(charge,int):
            self.charge = np.array([charge]*self.cars)
        elif isinstance(charge, np.ndarray):
            self.charge = charge
        else:
            raise ValueError('Incorrect charge type')
        
        if isinstance(charge_rate,int):
            self.charge_rate = np.array([charge_rate]*self.cars)
        elif isinstance(charge_rate, np.ndarray):
            self.charge_rate = charge_rate
        else:
            raise ValueError('Incorrecty charge_rate type')
        
        if len(self.charge) != self.cars:
            raise ValueError('Incorrect charge length')
        if len(self.charge_rate) != self.cars:
            raise ValueError('Incorrect charge_rate length')

        
    def update(self, actions, dt):
        """Updates charge rate using action sequence.

        Args:
            actions - numpy array of size (#cars, #actions)
            dt - time step to perform each action

        Returns:
            A tuple of cvxpy-type (variables, cost, constraints) to be passed 
            to a convex optimization solver
        
        Raises:
            ValueError if incorrect action shape
        """

        if actions.shape[0] != len(self.charge):
            raise ValueError('Incorrect actions length in update function')
        
        # add dimension if only performing one action
        if len(actions.shape)==1:
            actions = np.expand_dims(actions,1)

        for i in range(actions.shape[1]):
            self.charge = np.minimum(
                1, self.charge + self.charge_rate*actions[:,i]*dt)
            self.t += dt

    def optimization_problem(self, D, end_times, rise, eta, dt, relaxation=False):
        """Generates optimization problem.

        Args:
            D - A (2, # hours) numpy array, where the second row is the 
                expected value of base demand and the first row is the times 
                which those demands line up with.
            end_times - final time allowed to charge each car. Can be a single
                float, or a numpy array of length cars
            rise - individual demand rise per car
            eta - dual value multiplier (applied to electricity demand)
            dt - time step over which to perform each action
            relaxation - whether or not to form a relaxation

        Raises
        """

        if isinstance(end_times,(int,float)):
            end_times = np.array([end_times]*self.cars)
        elif isinstance(end_times, np.ndarray):
            pass
        else:
            raise ValueError('Incorrect end_times type')

        # total car numbers and action numbers
        c = self.cars
        a = round((end_times.max()-self.t)/dt)

        # generate true (interpolated) demand price curve
        times = np.arange(self.t, end_times.max(), dt)
        P = np.interp(times, D[0,:], D[1,:])

        # Define variable
        var = cp.Variable((c,a),boolean=not relaxation)

        # Define constraints
        constraints = []
        if relaxation:
            constraints += [var>=0]
            constraints += [var<=1]
        
        # Time constraints
        for i in range(a):
            for j in range(c):
                if self.t + dt*i >= end_times[j]:
                    constraints += [var[j,i]==0]

        # Total charge constraints
        for j in range(c):
            constraints += [
                self.charge[j]+dt*self.charge_rate[j]*cp.sum(var[j,:]) <= 1.0]
        
        # Calculate costs
        
        # electricity costs
        elec = 0
        for i in range(a):
            elec += cp.sum(var[:,i])*P[i]
            elec += rise*cp.quad_form(var[:,i],np.ones((c,c))) 
        elec *= dt

        # terminal costs
        terminal = 0
        for i in range(c):
            charge_i = self.charge[i]
            for j in range(a):
                charge_i += dt*self.charge_rate[i]*a[i,j]
            terminal += cp.inv_pos(charge_i)

        cost = eta*elec + terminal

        return var, cost, constraints
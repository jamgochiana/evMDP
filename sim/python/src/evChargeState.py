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

    def __init__(self, cars=10, charge=0.4, t=14, chargeRate=1./12):
        """Initializes EV charge state class."""

        self.t = t
        
    def update(self, actions, dt):
        """Updates charge rate using action sequence.

        Args:
            actions - numpy array of size (#cars, #actions)
            dt - time step to perform each action

        Raises
        """

        if actions.shape[0] != len(self.charge):
            raise ValueError('Incorrect actions length in update function')
        
        # add dimension if only performing one action
        if len(actions.shape)==1:
            actions = np.expand_dims(actions,1)

        for i in range(actions.shape[1]):
            self.charge = np.minimum(
                1, self.charge + self.chargeRate*actions[:,i]*dt)
            self.t += dt

    def update(self, P, endTimes, rise, lambda, dt):
        """Updates charge rate using action sequence.

        Args:
            actions - numpy array of size (#cars, #actions)
            dt - time step to perform each action

        Raises
        """


        return var, cost, constraints
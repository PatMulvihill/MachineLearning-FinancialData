"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""
"""Author: Lu Wang, lwang496, lwang496@gatech.edu"""

import numpy as np

# Author: Lu Wang lwang496
class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.Ttable = {}
        self.Qtable = np.random.uniform(-1, 1, (num_states, num_actions))
        self.r = np.ndarray((num_states, num_actions))
        self.r.fill(-1.0)

    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        rand_pick = np.random.random_sample()
        if  rand_pick>= self.rar:
            self.a = np.argmax(self.Qtable[s, :])
        else:
            self.a = np.random.randint(0, self.num_actions)

        self.s = s

        if self.verbose: print "s =", s, "a =", self.a
        return self.a

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """

        self.Qtable[self.s, self.a] = ((1 - self.alpha) * self.Qtable[self.s, self.a]) + self.alpha * (r + self.gamma * np.max(self.Qtable[s_prime, ]))

        if self.dyna != 0:
            a = self.a
            s = self.s

            self.r[s][a] = (1 - self.alpha) * (self.r[s][a]) + self.alpha * r
            if self.Ttable.get((s, a)) == None:
                self.Ttable[(s, a)] = []
            self.Ttable[(s, a)].append(s_prime)

            for i in range (self.dyna):
                s_r = np.random.randint(0, self.num_states)
                a_r = np.random.randint(0, self.num_actions)
                if self.Ttable.get((s_r, a_r)) != None:
                    s_p = np.random.randint(0, len(self.Ttable[(s_r, a_r)]))
                    s_n = self.Ttable[(s_r, a_r)][s_p]
                else:
                    s_n = np.random.randint(0, self.num_states)

                self.Qtable[s_r, a_r] = ((1 - self.alpha) * self.Qtable[s_r, a_r]) + self.alpha * (self.r[s_r][a_r] + self.gamma * np.max(self.Qtable[s_n,]))

        self.rar *= self.radr
        self.s = s_prime
        pick = np.random.random_sample()
        if pick>=self.rar:
            self.a = np.argmax(self.Qtable[self.s, :])
        else:
            self.a = np.random.randint(0, self.num_actions)

        if self.verbose: print "s =", s_prime, "a =", self.a, "r =", r
        return self.a

    def author(self):
        return 'lwang496'

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"

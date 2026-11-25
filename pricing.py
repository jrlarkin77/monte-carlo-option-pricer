import numpy as np
from scipy.stats import norm


def black_scholes_call(S0, K, T, r, sigma):

    """
    Calculates the Black-Scholes price for a European call option.

    S0 - Current stock price
    K - Strike price
    T - Time to expiration
    r - Risk-free rate
    sigma - Volatility

    Returns: 
        float: The Black-Scholes theoretical fair price for the European call option.
    """
    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))         #calculation for d1
    d2 = d1 - sigma*np.sqrt(T)                                              #calculation for d2
    price = S0*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)                   #Black-Scholes call formula: (fair price) = (expected benefit of holding stock) - (expected cost of paying strike price)
    return price


def monte_carlo_call(S0, K, T, r, sigma, n_sims):

    """
    Calculates the Monte Carlo fair price for a European call option.

    S0 - Current stock price
    K - Strike price
    T - Time to expiration
    r - Risk-free rate
    sigma - Volatility
    n_sims - Number of simulations

    Returns:
        float: The Monte Carlo estimated fair price for the European call option.
    """
    Z = np.random.standard_normal(n_sims)                                  #random draws shock
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)            #simulated final prices 
    payoffs = np.maximum(ST - K, 0)                                        #call option payoff
    price = np.exp(-r*T) * np.mean(payoffs)                                #discount back to today's value to get fair price
    return price


def antithetic_monte_carlo_call(S0, K, T, r, sigma, n_sims):

    """
    Calculates the Monte Carlo fair price for a European call option using antithetic variates for variance reduction

    S0 - Current stock price
    K - Strike price
    T - Time to expiration
    r - Risk-free rate
    sigma - Volatility
    n_sims - Number of simulations

    Returns:
        float: The Monte Carlo estimated fair price for the European call option using antithetic variates.
    """
    half_n = n_sims // 2                                                #cut simulaitons in half since antithetic generate pairs
    Z = np.random.standard_normal(half_n)                               #draw a random shock
    anti_Z = -Z                                                         #grab its opposite
    Z_tot = np.concatenate([Z, anti_Z])                                 #get full array of Z pairs
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z_tot)     #simulate final stock price for all Z's
    payoff = np.maximum(ST - K, 0)                                      #payoff for each simulation
    price = np.exp(-r*T) * np.mean(payoff)                              #discount back to today's value to get fair price
    return price


def black_scholes_put(S0, K, T, r, sigma):

    """
    Calculates the Black-Scholes price for a European put option.

    S0 - Current stock price
    K - Strike price
    T - Time to expiration
    r - Risk-free rate
    sigma - Volatility

    Returns:
        float: The Black-Scholes theoretical fair price for the European put option.
    """

    d1 = (np.log(S0/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))     #calculate d1
    d2 = d1 - sigma*np.sqrt(T)                                          #calculate d2
    price = K*np.exp(-r * T)*norm.cdf(-d2) - S0*norm.cdf(-d1)           #Black-Scholes put formula: (fair price) = (expected cost of paying strike price) - (expected benefit of holding stock)             
    return price


def monte_carlo_put(S0, K, T, r, sigma, n_sims):

    """
    Calculates the Monte Carlo fair price for a European put option.

    S0 - Current stock price
    K - Strike price
    T - Time to expiration
    r - Risk-free rate
    sigma - Volatility
    n_sims - Number of simulations

    Returns:
        float: The Monte Carlo estimated fair price for the European put option.
    """

    Z = np.random.standard_normal(n_sims)                                               #random draws shock 
    ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*Z)                         #simulated final prices
    payoff = np.maximum(K - ST, 0)                                                      #put option payoff ((K - ST) instead of (ST - K))
    price = np.exp(-r*T) * np.mean(payoff)                                              #discount back to today's value to get fair price
    return price


def sim_paths(S0, T, r, sigma, n_steps, n_sims):

    """
    Simulates stock price paths using the Geometric Brownian Motion model.

    S0 - Current stock price
    T - Time to expiration
    r - Risk-free rate
    sigma - Volatility
    n_steps - Number of time steps
    n_sims - Number of simulations

    Returns:
        numpy.ndarray: A matrix of simulated stock price paths.
    """

    dt = T / n_steps                                    #splits the time interval into steps
    paths = np.zeros((n_sims, n_steps + 1))             #makes a 0 matrix to store the simulated paths
    paths[:, 0] = S0                                    #fills first row with current stock price

    for t in range(1, n_steps + 1):                                                             #loops through each time step, ignore first column and go to expiration
        Z = np.random.standard_normal(n_sims)                                                   #generate random shocks
        paths[:, t] = paths[:, t - 1] * np.exp((r - 0.5 * sigma**2)*dt + sigma*np.sqrt(dt)*Z)   #fill each path with values step by step

    return paths


def LSM_put(S0, K, T, r, sigma, n_steps, n_sims, return_european=False):

    """
    Uses Longstaff-Schwartz method to price an American put option using Monte Carlo simulation.

    S0 - Current stock price
    K - Strike price
    T - Time to expiration
    r - Risk-free rate
    sigma - Volatility
    n_steps - Number of time steps
    n_sims - Number of simulations

    Returns:
        float: The estimated fair price for the American put option.
    """

    dt = T / n_steps                                        #divide time into smaller intervals
    paths = sim_paths(S0, T, r, sigma, n_steps, n_sims)     #simulate paths

    payoff = np.maximum(K - paths, 0)                       #get payoff at each step for all paths

    cash_flows = payoff[:, -1].copy()                       #grabs last column in payoff matrix (payoff at expiration)
    cash_flows_t = np.full(n_sims, n_steps)                 #initialize time of exercise for each simulation (assumes every option is exercised at expiration)

    for t in range(n_steps - 1, 0, -1):                     #loop backward starting at second to last step
        itm = payoff[:, t] > 0                              #boolean array for each time step to see if the option is currently in the money

        if np.sum(itm) == 0:                                #skips time step if no paths are in the money
            continue

        X = paths[itm, t]                                   #stock prices at time t for in-the-money paths

        passed_periods = cash_flows_t[itm] - t                      #grabs every paths currently known excercise time and finds steps away from current t
        Y = cash_flows[itm] * np.exp(-r * dt * passed_periods)      #discounts the payoff to the values of the current time step being evaluated

        #fits quadratic regression through scatter plot of stock price vs discounted cash flow to get expected value
        coeffs = np.polyfit(X, Y, deg=2)
        cont_value = np.polyval(coeffs, X)

        ex_value = payoff[itm, t]                                   #value if exercised at time t

        ex_now = ex_value > cont_value                              #decides if should be exercised

        index = np.where(itm)[0][ex_now]                            #list of paths where we should exercise at time t
        cash_flows[index] = payoff[index, t]                        #grabs payoff of each of those paths and overwrites the corresponding cash flow value
        cash_flows_t[index] = t                                     #update the time when the option should be exercised

    american = np.mean(cash_flows * np.exp(-r * dt * cash_flows_t))
    european = np.exp(-r * T) * np.mean(payoff[:, -1])  #euro put form same paths

    if return_european:
        return american, european
    return american
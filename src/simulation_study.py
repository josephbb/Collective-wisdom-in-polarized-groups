import numpy as np
import itertools as it
import pandas as pd

def ilogit(x):
    return 1/(1+np.exp(-x))

def get_network(politics, p):
    N = politics.shape[0]
    leaning_order = np.argsort(politics).argsort() / ((N - 1) / 5.0)
    side = 1.0 * (leaning_order + np.random.normal(0, 0.0000001)
                  > 2.5)  # Indicates on left side

    network = np.repeat(side, N).reshape((N, N))

    network[network == 0] = 1 - p
    network[network == 1] = p

    network[:, side == 0] = 1 - network[:, side == 0]
    probabilities = network / np.sum(network, axis=1)

    network[:, :] = 0
    n_connections = np.round(np.random.uniform(1, 15, N) / 2).astype('int')

    initiating_node = np.repeat(np.arange(N),n_connections)
    initiating_node_size = np.repeat(side, n_connections)

    left_connections = np.random.choice(np.arange(N),
                     size=initiating_node.size,
                     replace=True,
                     p = np.abs(side-p)/N*2)

    right_connections = np.random.choice(np.arange(N),
                     size=initiating_node.size,
                     replace=True,
                     p = np.abs(np.abs((1-side))-p)/N*2)
    connections = np.vstack([left_connections, right_connections])[initiating_node_size.astype('int'),
                                                             np.arange(initiating_node.size)]

    network[initiating_node, connections] = 1
    network = (network + network.T) > 0


    return network, leaning_order


def get_correct(diff, politics, lean, politics_shifted):
    N = politics.shape[0]
    if lean == 'False' or lean == False:
        correct = np.random.binomial(1, diff, politics.shape[0])
    elif lean == 'right':
        n_correct = np.random.binomial(N, diff)
        correct = np.zeros(N)
        correct[politics_shifted.argsort()[:n_correct]] = 1
    elif lean == 'left':
        n_correct = np.random.binomial(N, diff)
        correct = np.zeros(N)
        correct[politics_shifted.argsort()[-n_correct:]] = 1
    return correct



def get_belief(correct, diff, politics, belief_samps):
    N = politics.shape[0]
    chain = np.random.choice(1000, N)

    mu = ilogit(belief_samps['mu_a_pol'][chain, politics] + belief_samps['mu_b_pol'][chain, politics]*correct)
    alpha = ilogit(belief_samps['alpha_a_pol'][chain, politics] + belief_samps['alpha_b_pol'][chain, politics]*correct)
    gamma = ilogit(belief_samps['gamma_a_pol'][chain, politics] + belief_samps['gamma_b_pol'][chain, politics]*correct)
    theta = belief_samps['theta'][chain, politics]
    mu = np.random.beta(mu*theta, (1-mu)*theta)
    aa = np.random.binomial(1,alpha)
    gg = np.random.binomial(1,gamma)

    yhat = convert_to_confidence(aa * gg + (1-aa) * mu)
    return(yhat, chain)





def get_social_info(belief, network):
    masked = np.ma.masked_array(belief*network, mask=network ==0 )
    return masked.mean(axis=1)

def switching(belief, politics, social_info, diff, correct, extracted_switch_incorrect, extracted_switch_correct, chain_length=4000):
    x1 = (belief - 50) / 50 - .5


    x2 = social_info / 100 - .5


    N = politics.size
    chain = -np.random.choice(np.arange(chain_length), N)

    c_switch_theta = ilogit(extracted_switch_correct['alpha_p'][chain, politics] +
                                extracted_switch_correct['b_conf_p'][chain, politics] * x1 +
                                extracted_switch_correct['b_socConf_p'][chain, politics] * x2)

    i_switch_theta = ilogit(extracted_switch_incorrect['alpha_p'][chain, politics] +
                                extracted_switch_incorrect['b_conf_p'][chain, politics] * x1 +
                                extracted_switch_incorrect['b_socConf_p'][chain, politics] * x2)

    theta = correct * c_switch_theta + (1 - correct) * i_switch_theta
    switched = np.random.binomial(1, theta)
    return switched

def run_single(p, diff, N, proportion, lean,belief_samps,extracted_switch_incorrect, extracted_switch_correct):
    proportion = proportion / np.sum(proportion).astype('float')
    politics = np.random.choice(np.arange(5), N, p=proportion)
    network, politics_shifted = get_network(politics, p)
    correct = get_correct(diff, politics, lean, politics_shifted)

    #Get individual beliefs
    pcopy = politics.copy()
    belief, chain = get_belief(correct, diff, pcopy, belief_samps)
    bcopy = belief.copy()
    bcopy[correct == 1] = np.abs(100-bcopy[correct==1])
    social_info = get_social_info(bcopy, network)
    ##Convert belief to reported confidence scale
    belief = np.abs(belief-50)+50
    switched = switching(belief, politics, social_info,diff, correct, extracted_switch_incorrect, extracted_switch_correct)
    correct_final = (correct + switched) % 2

    return [np.mean(correct_final), np.mean(correct)]

def run_simulations(dd, belief_samps, extracted_switch_correct, extracted_switch_incorrect):
    allNames = sorted(dd)
    combinations = it.product(*(dd[Name] for Name in allNames))
    dat = pd.DataFrame(list(combinations), columns=allNames)

    dat['correct_final'], dat['correct_start'] = np.vectorize(run_single)(dat['p'],
                                                                         dat['diff'],
                                                                         dat['N'],
                                                                         dat['proportions'],
                                                                         dat['lean'],
                                                                         belief_samps,
                                                                         extracted_switch_incorrect,
                                                                         extracted_switch_correct)

    return dat


def convert_to_confidence(x):
    return 100*(x/2 + .5)

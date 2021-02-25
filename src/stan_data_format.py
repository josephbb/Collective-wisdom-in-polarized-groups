import numpy as np
import pandas as pd
from scipy.stats import rankdata
import patsy as pt
def scale(t):
    return (t-np.mean(t))/np.std(t)


def recode_id(df):
    """Recode ID's from 1:N"""
    data = df.copy()
    id_recode = [dict(zip(np.unique(data.id),
                          np.arange(np.unique(data.id).shape[0])))[ids] for ids in data.id.values]
    data['id_recode'] = id_recode
    return data

def get_pol_by_id(df):
    """Return a list of political leanings in order of recoded_ids"""
    data = df.copy()
    pols = []
    for idx in np.unique(data['id_recode']):
        pols.append(int(np.mean(data['pol_recode'][data['id_recode']==idx])))
    return pols

def format_hierarchichal(df, answer=False):
    """Separate out data for false questions, recode_ids and gather political leanings"""
    false_data = df[df.answer==answer]
    #Recode ID's from 1:N
    false_data = recode_id(false_data)
    #Gather political leaning by ID
    pols = get_pol_by_id(false_data)
    return false_data, pols

def format_stan_data_logistic(exp_1_data):
    """Formats data for the accuracy/confidence model in experiment 1"""
    #Recode ID and gather political leanings by ID
    false_data, pols = format_hierarchichal(exp_1_data, answer=False)
    #Create a dictionary for the data.
    stan_data_logistic = dict(correct = false_data['correct_start'].astype('int').values,
                        j = np.unique(false_data['id_recode']).shape[0],
                        n = false_data.shape[0],
                        P = 5,
                        state = false_data['state_recode'].values+1,
                        NStates = np.unique(false_data['state_recode']).shape[0],
                        pol = np.array(pols),
                        x =  2*((false_data['numConf'].values-50)/50-.5),
                        who = false_data['id_recode']+1)
    return stan_data_logistic

def format_stan_data_belief(exp_1_data):
    """Formats data for the agent/belief model in the simulation study"""
    #Seaprate False, Recode ID and gather political leanings by ID
    false_data, pols = format_hierarchichal(exp_1_data, answer=False)
    #Return dictionary
    stan_data = dict(correct = false_data['correct_start'].astype('int').values,
                        j = np.unique(false_data['id_recode']).shape[0],
                        n = false_data.shape[0],
                        P = 5,
                        pol = np.array(pols),
                        y = (false_data['numConf'].values-50)/50,
                        who = false_data['id_recode']+1)

    return stan_data

def calculate_switching(df):
    data = df.copy()
    switched = df.response.values != df.socialAnswer.values
    switched = 1*switched
    data['switched'] = switched
    return data



def format_stan_data_switch(exp_1_data, correct=False):
    """Formats data for the correct/incorrect switching model"""
    #Subset False Data
    false_data = exp_1_data[exp_1_data.answer==False]

    #Determine if they switced (changed their guess)
    false_data = calculate_switching(false_data)

    #Subset those that are correct/incorrect
    df = false_data[false_data.correct_start==correct]

    #Format the data for the hierarchichal model_location
    df, pols = format_hierarchichal(df, answer=False)
    stan_model_data = dict(n=df.shape[0],
                           y=df['switched'],
                           confidence = (df['numConf'].values-50)/50-.5,#Scale a bit
                           socConfidence = df.socialInfo/100-.5,#Scale a bit
                           who = df['id_recode'].values+1,
                           j = np.unique(df['id_recode']).shape[0],
                           pol=pols)
    return stan_model_data, df


def format_stan_data_exp2(exp_2_data):
    false_exp2_data = exp_2_data[exp_2_data.answer==False]
    N = false_exp2_data.shape[0]
    p = 1*(false_exp2_data.p_recode.values)
    cond = false_exp2_data['cond_recode'].values+1
    y = 1*(false_exp2_data['social_correct'].values)
    false_exp2_data['cond_recode']
    pd.Categorical(false_exp2_data['cond_recode'])
    grouped_exp2 = false_exp2_data.groupby(['state', 'cond_recode', 'p_recode']).mean().reset_index()


    state_data = dict(y=grouped_exp2['social_correct'].values,
                        N = grouped_exp2.shape[0],
                        p = grouped_exp2['p_recode'].values.astype('int'),
                        cond = grouped_exp2['cond_recode'].values+1)
    return state_data, false_exp2_data

def format_stan_data_exp3(melted,with_cb=False):
    ind_predictors = melted.groupby('id_recode').mean().reset_index()
    ind_predictors['pol_recode'] = pd.Categorical(ind_predictors['pol_recode'].astype('int'))
    melted['pol_recode2'] = pd.Categorical(melted['pol_recode'])
    melted['question_code2'] = pd.Categorical(melted['question_code'])

    t =ind_predictors['BSS'].values
    for var in ['BSS','WSS','EIS','HBS','NUMS','RIS']:
        ind_predictors[var+'_s'] = scale(ind_predictors[var].values)
    if with_cb:
        _, u = pt.dmatrices('correct~BSS_s+WSS_s+EIS_s+HBS_s+NUMS_s+RIS_s-1',ind_predictors)
    else:
        _, u = pt.dmatrices('correct~C(pol_recode)-1',ind_predictors)
    u = np.array(u)

    _, x1 = pt.dmatrices('correct~C(pol_recode)+C(pol_recode):conf-1',melted)
    x1 = np.array(x1)
    x2 = np.vstack([np.ones(melted.shape[0]), scale(melted['conf'].values)])

    exp3_stan_df = dict(N=x2.shape[1],
                        K2=x2.shape[0],
                        K1=x1.shape[1],
                        Q=melted['question_code'].unique().size,
                        S=melted['id_recode'].unique().size,
                        L = u.shape[1],
                        y=melted['correct'].astype('int').values,
                        ss = melted['id_recode'].values,
                        qq=melted['question_code'].values,
                        x2=x2.T,
                        x1=x1,
                        conf=melted['conf'].values,
                        u =u.T)
    return exp3_stan_df
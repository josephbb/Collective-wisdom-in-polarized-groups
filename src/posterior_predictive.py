import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

def convert_to_confidence(x):
    return 100*(x/2 + .5)

def plot_predicted_vs_observed_belief_model(belief_samps,exp_1_data,q=[5.5,94.5]):
    #Add posterior and posterior mean to dataframe
    false_data = exp_1_data[exp_1_data['answer']==False]
    false_data['yhat'] = belief_samps['yhat'][-1,:]
    false_data['mean_yhat'] = np.mean(false_data['yhat'],axis=0)

    #Define palletes, one with a grey center.
    pal1 = sns.diverging_palette(10, 220, sep=80, n=5,l=30,center='light')
    pal2 = sns.diverging_palette(10, 220, sep=80, n=5,l=10,center='dark')

    #Calculate credible regions and mean, convert to confidence (from scaled)
    lower = np.zeros(5)
    upper = np.zeros(5)
    mean = np.zeros(5)

    for idx in range(5):
        lower[idx], upper[idx] = np.percentile(np.mean(belief_samps['yhat'][:, false_data['pol_recode']==idx+1],axis=1),
                                        q)
        mean[idx] =  np.mean(belief_samps['yhat'][:, false_data['pol_recode']==idx+1])

    np.vstack([lower,upper]).shape
    lower = convert_to_confidence(lower)
    upper = convert_to_confidence(upper)
    mean = convert_to_confidence(mean)


    #Plot expected vs. observed political-group level data
    observed = false_data.groupby('pol_recode').mean()['numConf'].values
    for idx in range(5):
        if idx !=2:
            color = pal1[idx]
        else:
            color = 'grey'
        plt.scatter(mean[idx], observed[idx],color=color)
        plt.plot([mean[idx], mean[idx]], [lower[idx], upper[idx]],color=color)
    plt.plot([50,100],[50,100],ls='--', color='k')


    plt.ylim(85,95)
    plt.xlim(85,95)
    plt.yticks(np.linspace(85,95,6).astype('int'))
    plt.xticks(np.linspace(85,95,6).astype('int'))
    plt.xlabel('Predicted')
    plt.ylabel('Observed')
    return plt.gcf()

def plot_belief_model_distributions(exp_1_data, belief_samps):
    false_data = exp_1_data[exp_1_data['answer']==False]
    false_data['yhat'] = belief_samps['yhat'][-1,:]
    false_data['mean_yhat'] = np.mean(false_data['yhat'],axis=0)

    pal1 = sns.diverging_palette(10, 220, sep=80, n=5,l=30,center='light')
    sns.set_palette(pal1)
    for idx in range(5):
        plt.subplot(2,5,idx+1)
        if idx == 0:
            plt.ylabel('Density \n (Correct)')
        color = pal1[idx]
        temp = false_data[false_data['pol_recode']==idx+1]
        temp = temp[temp.correct_start==True]
        ax = plt.gca()

        sns.kdeplot(temp['numConf'],color=color,shade=True,alpha=1,legend=False,clip=[49,101])
        sns.kdeplot(convert_to_confidence(temp['yhat']),color='grey',shade=True,legend=False,clip=[49,101])
        plt.xlim(50,100)
        if idx > 0:
            plt.yticks([])

    for idx in range(5):

        plt.subplot(2,5,5+idx+1)
        if idx == 0:
            plt.ylabel('Density \n (Incorrect)')
        color = pal1[idx]
        temp = false_data[false_data['pol_recode']==idx+1]
        temp = temp[temp.correct_start==False]
        ax = plt.gca()

        sns.kdeplot(temp['numConf'],color=color,shade=True,alpha=1,legend=False,clip=[49,101])
        sns.kdeplot(convert_to_confidence(temp['yhat']),color='grey',shade=True,legend=False,clip=[49,101])
        plt.xlim(50,100)
        if idx > 0:
            plt.yticks([])
    plt.tight_layout()

import patsy
import seaborn as  sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import expit

def plot_posterior_predictive_exp3(melted,samples)
    pal = sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='light')
    pal2 = sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='dark')
    pal[2] = np.array(pal2[2])*4
    plt.figure(figsize=(4,4))

    samples['y_hat']
    for pidx in range(5):
        for question in melted['question_code'].unique():
            indices = (melted['question_code']==question) & (melted['pol_recode']==pidx+1)
            y1 = np.mean(samples['y_hat'][:,indices])
            x1 = np.mean(melted['correct'][indices])
            plt.scatter(y1,x1,color=pal[pidx],alpha=.5)
            plt.plot()
    plt.plot([0,1], [0,1], ls='--',color='grey')
    plt.ylabel('Predicted accuracy')
    plt.xlabel('Fitted accuracy')
    plt.xlim(0,1)
    plt.ylim(0,1)
    
def plot_figure4c(samples):
    pal = sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='light')
    pal2 = sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='dark')
    pal[2] = pal2[2]
    sns.set_context('paper', font_scale=1.5)
    plt.figure(figsize=(3,3))

    for pol in range(5):
        vals = []
        for ii in range(50):
            chain = np.random.choice(4000)
            u = np.zeros(11)
            x = np.zeros(5)
            u[pol] = 1
            x[pol] = 1
            xtemp = np.linspace(0,1,100)

            temp = samples['alpha'][chain]
            temp +=samples['beta'][chain]*xtemp
            temp +=np.sum(np.mean(samples['beta_q'][chain,:5,:,],axis=1).T*x)
            temp +=np.sum(np.mean(samples['beta_q'][chain,5:,:],axis=1).T*x)*xtemp 
            temp +=np.sum(samples['gamma'][chain,0,:]*x)
            temp +=np.sum(samples['gamma'][chain,1,:]*x)*xtemp
            #temp += np.sum(samples['gamma'][chain,0,:])*x
            vals.append(expit(temp))
        mean = np.mean(np.vstack(vals),axis=0)
        plt.plot(xtemp*50+50, mean, color=pal[pol])
        ci =  np.percentile(np.vstack(vals), q=[5.5,94.5],axis=0)
        plt.fill_between(xtemp*50+50, ci[0], ci[1],alpha=.3,color=pal[pol])
    plt.ylim(0,1)
    plt.xlim(50,100)

    plt.ylabel('Accuracy')
    plt.xlabel('Reported confidence \n(domain controlled)')
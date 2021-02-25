import patsy
import seaborn as  sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.special import expit


pal = sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='light')
pal2 = sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='dark')
pal[2] = pal2[2]

    
def plot_posterior_predictive_exp3(melted,samples):
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
    plt.ylabel('Observed accuracy')
    plt.xlabel('Predicted accuracy')
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
def plot_exp3_demographics(data):    
    #Set some basic plotting parameters. 
    sns.set_style('white')
    sns.set_context('paper',font_scale=1.5)



    #Plot age demographics
    plt.figure(figsize=(6,6))
    plt.subplot(4,1,1)
    sns.countplot(y=data.DEM_1,order=['18-24','25-34','35-44','45-54','55-64','65+'],color='grey')
    plt.xlim(0,120)
    plt.ylabel('')
    plt.xlabel('')

    #Plot gender demographics
    plt.subplot(4,1,2)
    sns.countplot(y=data.DEM_4,order=['Male','Female','Other/Prefer not to say'],color='grey')
    plt.ylabel('')
    plt.xlabel('')

    #Plot education demographics
    plt.subplot(413)
    sns.countplot(y=data.DEM_3,order=['Some High School', 
                  'High School',
                  'Some College',
                  'College', 
                  'Graduate Degree or Higher'],color='grey')
    plt.ylabel('')
    plt.xlabel('')

    #plt.savefig('../../Graphs/Edu2.pdf',fmt='pdf',dpi=1500,transparent=True)

    #Plot political leaning demographics
    plt.subplot(4,1,4)
    sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='light'))
    sns.countplot(y=data.DEM_2,order = ['Very Conservative','Conservative',
                                    'Moderate','Liberal','Very Liberal'])
    plt.xlim(0,120)
    plt.ylabel('')
    plt.tight_layout()
    
def plot_exp3_incidental(melted, samples):
    order = melted.groupby(['question_code']).mean()['correct'].argsort()

    q_names = []
    for item in melted.groupby(['question_code']):
        q_names.append(item[1]['question'][0])


    plt.figure(figsize=(6,8))

    plt.subplot(121)
    for yp,idx in enumerate(order):
        for jdx in range(5):
            temp = np.mean(samples['beta_q'][:,jdx+5,idx],axis=0)
            ci = np.percentile(samples['beta_q'][:,jdx+5,idx],q=[5.5,94.5])
            plt.scatter(temp,3*(yp+((jdx-3)*.1)),color=pal[jdx],alpha=.3)
            plt.plot([ci[0], ci[1]], [3*(yp+((jdx-3)*.1)),3*(yp+((jdx-3)*.1))],color=pal[jdx],alpha=.7)

    plt.yticks(np.arange(order.size)*3, np.array(q_names)[order])
    #plt.ylabel('alpha')
    plt.xlabel('Effect of confidence \nby question')
    plt.plot([0,0],[-1,3*np.max(order)+1],ls='--',color='k')
    plt.ylim(-2, 3*np.max(order)+3)

    plt.subplot(122)
    jdx = 2
    for yp,idx in enumerate(order):
        temp1 = samples['beta_q'][:,5,idx]
        temp2 = samples['beta_q'][:,-1,idx]
        ci = np.percentile(temp1-temp2,q=[5.5,94.5])
        plt.scatter(np.mean(temp1-temp2),3*(yp+((jdx-3)*.1)),color='k',alpha=.3)
        plt.plot([ci[0], ci[1]], [3*(yp+((jdx-3)*.1)),3*(yp+((jdx-3)*.1))],color='k',alpha=.7)

    #plt.yticks(np.arange(order.size)*3, np.array(q_names)[order])
    #plt.ylabel('alpha')
    plt.yticks([])
    plt.xlabel('VC-VL effect of \nconfidence')
    plt.plot([0,0],[-1,3*np.max(order)+1],ls='--',color='k')
    plt.ylim(-2, 3*np.max(order)+3)


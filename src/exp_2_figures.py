import patsy
import seaborn as  sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

pal = sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='light')
pal2 = sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='dark')
pal[2] = pal2[2]
def ilogit(x):
    return 1/(1+np.exp(-x))

def plot_state_changes(grouped_exp2, exp2_samples):
    pal=sns.color_palette('Greens', n_colors=2)


    for idx in range(2):
        temp = grouped_exp2[grouped_exp2['p_recode']==idx]
        for ii in range(3):
            mean = np.mean(ilogit(exp2_samples['alpha'][:,ii] + exp2_samples['beta'][:,ii]*idx))
            cr = np.percentile(ilogit(exp2_samples['alpha'][:,ii] + exp2_samples['beta'][:,ii]*idx), q=[5.5, 94.5])

            plt.scatter(ii+idx*.3, mean, s=150, color=pal[idx])
            plt.plot([ii+idx*.3, ii+idx*.3],
                     cr,lw=8, color=pal[idx],alpha=.8)
        plt.scatter(temp['cond_recode']+idx*.3, temp['social_correct'], alpha=.8,color='k')
    plt.xlim(-.33,2.66)
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(3) + np.arange(3)*.15,['Nonpartisan', 'Liberal\ncorrect', 'Conservative \ncorrect'])


def plot_fig2E(exp2_data, exp2_samples):
    grouped_exp2 = exp2_data.groupby(['state', 'cond_recode', 'p_recode']).mean().reset_index()
    pal=sns.color_palette('Greens', n_colors=5)
    pal = [pal[2], pal[4]]

    for jj in range(3):
        temp = grouped_exp2[grouped_exp2['cond_recode']==jj]
        y = np.hstack([temp[temp['p_recode']==False]['social_correct'].values,
                        temp[temp['p_recode']==True]['social_correct'].values])
        x = np.hstack([np.repeat(jj,10), np.repeat(jj + .3,10)])

        for k in range(10):
            plt.plot([x[k], x[k+10]],[y[k],y[k+10]] ,color='grey')

    for idx in range(2):
        temp = grouped_exp2[grouped_exp2['p_recode']==idx]
        for ii in range(3):
            mean = np.mean(ilogit(exp2_samples['alpha'][:,ii+3*idx]) )
            cr = np.percentile(ilogit(exp2_samples['alpha'][:,ii+3*idx]), q=[5.5, 94.5])

            plt.scatter(ii+idx*.3, mean, s=150, color=pal[idx])
            plt.plot([ii+idx*.3, ii+idx*.3],
                     cr,lw=8, color=pal[idx],alpha=.8)
        plt.scatter(temp['cond_recode']+idx*.3, temp['social_correct'], alpha=.8,color='k')



    ax = plt.gca()
    mean_p5 = np.mean(ilogit(exp2_samples['alpha'][:,2]))
    mean_p98 = np.mean(ilogit(exp2_samples['alpha'][:,5]))
    print(mean_p5)

    head_length = .02
    ax.arrow(2.3+idx*.3/2,mean_p5, 0,mean_p98-mean_p5 + head_length,
                head_width=0.1, head_length=head_length, fc='grey', ec='grey')


    plt.xlim(-.33,2.66)
    plt.ylabel('Accuracy')
    plt.xticks(np.arange(3) + np.arange(3)*.15,['Nonpartisan', 'Liberal\ncorrect', 'Conservative \ncorrect'])


def plot_fig2F(exp2_samples):

    pal = sns.diverging_palette(10, 220, sep=80, n=3,l=40,center='light')
    pal2 = sns.diverging_palette(10, 220, sep=80, n=3,l=40,center='dark')
    pal[1] = pal2[1]

    pal_order = [2,1,0]


    for idx in range(3):
        sns.kdeplot(ilogit(exp2_samples['alpha'][:,3+idx])-ilogit(exp2_samples['alpha'][:,idx]),
                    shade=True,
                    color=pal[pal_order[idx]])



    plt.xlabel('Impact of homophily\non accuracy')
    plt.ylabel('Density')

##Here we plot distributrions of how many individuals were correct for each states.
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

pal = sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='light')
pal2 = sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='dark')
pal[2] = pal2[2]
def ilogit(x):
    return 1/(1+np.exp(-x))

def plot_figure1a(true_data, false_data):

    #Basic figure paramters
    sns.set_context('paper', font_scale=1.5)

    #Plot distributions, adjust legend etc...
    sns.distplot(true_data.groupby(['states']).mean()['correct_start'],hist_kws=dict(histtype='stepfilled',alpha=.9,ec="k"),
                 color='white',bins=np.linspace(0,1,10),label='True',kde=False)
    sns.distplot(false_data.groupby(['states']).mean()['correct_start'],hist_kws=dict(histtype='stepfilled',alpha=.8,ec="k"),
                 color='grey',bins=np.linspace(0,1,10),label='False',kde=False)
    plt.yticks(np.linspace(0,25,6))
    plt.xlim(0,1)
    plt.xlabel('Proportion correct')
    plt.ylabel('Count')


    #Save figure
    plt.tight_layout()

def joint_hpdi(samples_extracted):

    for idx in range(5):
        x = samples_extracted['alpha_p'][:,idx]
        y = samples_extracted['beta_p'][:,idx]
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        #set zi to 0-1 scale
        zi = (zi-zi.min())/(zi.max() - zi.min())
        zi =zi.reshape(xi.shape)

        #set up plot
        origin = 'lower'
        levels = [.11,1]

        CS = plt.contourf(xi, yi, zi,levels = levels,
                      shade=True,
                      linewidths=(1,),
                      alpha=.5,
                      colors=[pal[idx], pal[idx]],
                      origin=origin)
    plt.xlabel('Intercept')
    plt.ylabel('Effect of \nconfidence')
    plt.ylim(-1.5,1.5)
    plt.xlim(-1,1)
    plt.xticks(np.linspace(-1.5,1.5,5))
    plt.xticks(np.linspace(-1.5,1.5,5))





def plot_figure1b(samples_extracted,stan_data_logistic):
    x = np.linspace(np.min(stan_data_logistic['x']),np.max(stan_data_logistic['x']),10)


    for idx in range(5):
        y = np.array([samples_extracted['alpha_p'][:,idx] + samples_extracted['beta_p'][:,idx] * item for item in x])
        y = ilogit(y)
        cis = np.percentile(y, q=[5.5,94.5],axis=1)
        plt.plot(50*(x/2+.5)+50, np.mean(y, axis=1),color=pal[idx])
        plt.fill_between(50*(x/2+.5)+50, cis[0,:], cis[1,:],alpha=.3,color=pal[idx])
    plt.ylim(.2,.8)
    plt.xlim(50,100)
    plt.ylabel('Accuracy')
    plt.xlabel('Reported confidence')

def plot_fig1cd(stan_model_data, df, samples, correct=True):
    x = np.linspace(-.5, .5, 100)

    x_transformed = (x+.5)*100


    for idx in range(5):
        avg_conf =  np.mean(stan_model_data['confidence'][df['pol_recode']==idx+1])

        y = np.array([ilogit(samples['alpha_p'][:,idx] + \
                             samples['b_conf_p'][:,idx] * avg_conf +\
                             samples['b_socConf_p'][:,idx] * item) for item in x])
        if correct:
            plt.plot(x_transformed, np.mean(y,axis=1),color=pal[idx])
            ci = np.percentile(y, axis=1, q=[5.5,94.5])
            plt.fill_between(x_transformed, ci[0], ci[1], color=pal[idx],alpha=.3)
        else:
            plt.plot(x_transformed[::-1], np.mean(y,axis=1),color=pal[idx])
            ci = np.percentile(y, axis=1, q=[5.5,94.5])
            plt.fill_between(x_transformed[::-1], ci[0], ci[1], color=pal[idx],alpha=.3)
    plt.ylabel('Probability of switching')
    plt.ylim(0,1)
    plt.xlim(0,100)
    plt.xlabel('Social disagreement')



def plot_switch_predicted_acuracy(data, switch_samples, correct=True):
    extracted_switch_samples_correct = switch_samples.extract(['alpha_p',
                                            'b_conf_p',
                                            'b_socConf_p',
                                            'yhat'])
    correct_data = data[data['correct_start']==correct]

    pal[2] = pal2[2]
    sns.set_context('paper', font_scale=1.5)
    correct_data['yhat'] = np.mean(extracted_switch_samples_correct['yhat'],axis=0)

    grouped = correct_data.groupby(['pol_recode']).mean().reset_index()
    plt.scatter(grouped['yhat'], grouped['switched'],color=pal,s=100)
    plt.plot([0,1], [0,1], ls='--', color='black')
    plt.ylim(0.15, 0.4)
    plt.xlim(0.15, 0.4)
    plt.yticks(np.linspace(.15,.4,6))
    plt.yticks(np.linspace(.15,.4,6))
    plt.xlabel('Predicted \nswitching')
    plt.ylabel('Observed \nswitching')

    np.percentile(extracted_switch_samples_correct['yhat'],axis=1, q=[5.5, 94.5])

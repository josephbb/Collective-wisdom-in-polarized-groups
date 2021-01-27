import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
def demographics_exp_1(raw):
    """Plots demographics for experiment 1"""

    unparsed = raw
    #Set some basic plotting parameters.
    sns.set_style('white')
    sns.set_context('paper',font_scale=1.5)
    #Load the unparsed data

    #Plot age demographics
    plt.subplot(4,1,1)
    sns.countplot(y=unparsed.age,color='grey',order=['18-24','25-34','35-44','45-54','55-64','65+'])
    plt.ylabel('')
    plt.xlabel('')

    #Plot gender demographics
    plt.subplot(4,1,2)
    sns.countplot(y=unparsed.gender,color='grey')
    plt.ylabel('')
    plt.xlabel('')
    plt.subplot(4,1,3)

    #Plot education demographics
    sns.countplot(y=unparsed.education,color='grey',order=['Some High School','High School',
                                                          'Some College','College','Graduate Degree or Higher'])
    plt.ylabel('')
    plt.xlabel('')

    #Plot political leaning demographics
    plt.subplot(4,1,4)
    sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='light'))
    sns.countplot(y=unparsed.politics,
                  order=['Very Conservative','Conservative','Moderate','Liberal','Very Liberal'],)
    plt.ylabel('')
    plt.tight_layout()

    return plt.gcf()

def demographics_exp_2(raw):
    """Plots demographics for experiment 2"""
    #Load the unparsed data
    unparsed = pd.DataFrame(([item[1].iloc[0] for item in raw.groupby('cintID')]))

    #Plot age demographics
    plt.subplot(4,1,1)
    sns.countplot(y=unparsed.age,color='grey',order=['18-24','25-34','35-44','45-54','55-64','65+'])
    plt.ylabel('')
    plt.xlabel('')

    #Plot gender demographics
    plt.subplot(4,1,2)
    sns.countplot(y=unparsed.gender,color='grey')
    plt.ylabel('')
    plt.xlabel('')
    plt.subplot(4,1,3)

    #Plot education demographics
    sns.countplot(y=unparsed.education,color='grey',order=['Some High School','High School',
                                                          'Some College','College','Graduate Degree or Higher'])
    plt.ylabel('')
    plt.xlabel('')

    #Plot political leaning demographics
    plt.subplot(4,1,4)
    sns.set_palette(sns.diverging_palette(10, 220, sep=80, n=5,l=40,center='light'))
    sns.countplot(y=unparsed.politics,
                  order=['Very Conservative','Conservative','Moderate','Liberal','Very Liberal'],)
    plt.ylabel('')
    plt.tight_layout()

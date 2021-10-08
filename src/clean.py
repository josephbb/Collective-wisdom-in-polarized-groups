import pandas as pd
import numpy as np
def custom_melt(row, n_questions=20):
    response = [] #The answer they gave
    numConf = [] #Their raw reported confidence.
    qualConf = [] #Their qualitatively reported confidence.
    socialAnswer = [] #The social information they received.
    socialNumConf = [] #Their confidence after receiving social information.
    socialQualConf = [] #The qualitative ocnfidence after receiving social information.
    answers = [] #The correct answer.
    states=[] #The state .
    trueSocial = [] #The correct answer.
    socialSize = [] #The number of individuals they interacted with.

    #Loop through each question they recieved (20 question total) and determine what question they had an what answer they gave.
    for idx in range(n_questions):
        response.append(row['Q'+str(idx)])
        numConf.append(row['Q'+str(idx)+'S_1'])
        qualConf.append(row['Q'+str(idx)+'T'])
        socialAnswer.append(row['Q'+str(idx)+'Social'])
        socialNumConf.append(row['Q'+str(idx)+'SocialS_1'])
        socialQualConf.append(row['Q'+str(idx)+'SocialT'])
        answers.append(row['answer'+str(idx)])
        states.append(row['abbr'+str(idx)])
        trueSocial.append(row['trueVal'+str(idx)])
        socialSize.append(row['resp'+str(idx)])

    #STore their data as a dict.
    dataDict = {'response':response,
               'numConf':numConf,
               'qualConf':qualConf,
               'socialAnswer':socialAnswer,
               'socialNumConf':socialNumConf,
               'socialQualConf':socialQualConf,
               'answer':answers,
               'states':states,
               'id':row['CintID'],
               'gender':row['gender'],
               'politics':row['politics'],
               'education':row['education'],
               'age':row['age'],
               'socialSize':socialSize,
               'socialInfo':trueSocial} ##probability social information indicates correct answer
    #Make a list of dictionaries.
    return pd.DataFrame(dataDict)

def clean_exp_1(exp1_data_raw):
    "Tidies experiment 1 data"
    data_exp1 = pd.concat([item for item in exp1_data_raw.apply(custom_melt, axis=1)] )
    #Determine if the participants were correct, drop empty frames.

    data_exp1['correct_start'] = data_exp1.answer == data_exp1.response
    data_exp1 = data_exp1.dropna()

    #Split data into those where the correct answer is true and compute the proportion correct (difficulty)
    true_data = data_exp1[data_exp1.answer==True].copy()
    difficulty = dict(zip(true_data.groupby('states').mean().correct_start.keys(),
                          true_data.groupby('states').mean().correct_start.values))
    true_data['difficulty'] = [difficulty[item] for item in true_data.states.values]

    #Split data into those where the correct answer is false and compute the proportion correct (difficulty)
    false_data = data_exp1[data_exp1.answer==False].copy()
    difficulty = dict(zip(false_data.groupby('states').mean().correct_start.keys(),
                          false_data.groupby('states').mean().correct_start.values))
    false_data['difficulty'] = [difficulty[item] for item in false_data.states.values]

    #Merge those two
    data = pd.concat([true_data,false_data])

    #Use a dictionary to recode political leaning, gender, education, id and age
    pol_dict = {'Very Conservative':1, 'Conservative':2,'Moderate':3, 'Liberal':4,'Very Liberal':5}
    data['pol_recode'] = [pol_dict[item] for item in data.politics]


    gender_dict = {'Male':1, 'Female':2,'Other/Prefer not to say':3}
    data['gender_recode'] = [gender_dict[item] for item in data.gender]

    edu_dict = {'Graduate Degree or Higher':1 ,'College':2 ,'Some College':3 ,'High School':4,'Some High School':5}
    data['edu_recode'] = [edu_dict[item] for item in data.education]

    age_dict = {'18-24':1,'25-34':2,'35-44':3,'45-54':4,'55-64':5,'65+':6}
    data['age_recode'] = [age_dict[item] for item in data.age]

    id_recode = [dict(zip(np.unique(data.id), np.arange(np.unique(data.id).shape[0])))[ids] for ids in data.id.values]
    data['id_recode'] = id_recode

    state_recode = [dict(zip(np.unique(data.states), np.arange(np.unique(data.states).shape[0])))[state] for state in data.states.values]
    data['state_recode'] = state_recode
    #Save that to the preliminarily parsed data.

    return data

def clean_exp_2(dat):
    """Tidies experiment 2 data"""
    dat = dat.drop(columns=['Old'])
    dat = dat[dat['Finished']==True]


    dat = dat.dropna()
    dat.reset_index()

    recode = dict(zip(np.unique(dat['CintID'].values),
                    np.arange(np.unique(dat['CintID'].values).shape[0])+1))


    idx = 0
    aa = 0
    all_data = []
    for idx in dat.index.values:
        for ii in range(10):
            one_question_dict = {'cintID':dat.loc[idx]['CintID'],
             'id_recode':recode[dat.loc[idx]['CintID']],
             'sessionID':dat.loc[idx]['sessionid'],
             'Duration':dat.loc[idx]['Duration (in seconds)'],
            'question':dat.loc[idx]['questions.'+str(ii)],
            'state':dat.loc[idx]['questions.'+str(ii)].split('_')[0],
            'age':dat.loc[idx]['Q5'],
            'politics':dat.loc[idx]['Q7'],
            'education':dat.loc[idx]['Q9'],
            'gender':dat.loc[idx]['Q11'],
            'answer':dat.loc[idx]['questions.'+str(ii)].split('_')[1]=='T',
            'asocial_response':dat.loc[idx][str(ii+1)+'_Q43']==True,
            'asocial_conf':int(dat.loc[idx][str(ii+1)+'_Q46_1']),
            'asocial_qual':dat.loc[idx][str(ii+1)+'_Q48'],
            'TrueChance':dat.loc[idx]['TrueChance'+str(ii)],
            'neighbors':dat.loc[idx]['neighbors_n'+str(ii)].astype('int'),
            'p':dat.loc[idx]['p'+str(ii)],
            'condition':dat.loc[idx]['condition'+str(ii)],
            'social_response':dat.loc[idx]['q'+str(ii)+'Social']==True,
            'social_conf':dat.loc[idx]['conf'+str(ii)+'Social_1'].astype('int'),
            'social_qual':dat.loc[idx]['qual'+str(ii)+'Social'],
            'social_correct_ref':dat.loc[idx]['correct'+str(ii)]}

            if np.isnan(dat.loc[idx]['correct'+str(ii)]):
                aa+=1


            all_data.append(one_question_dict)


    df = pd.DataFrame(all_data)
    pol_recode_dict = {'Very Conservative':0,
                        'Conservative':1,
                      'Moderate':2,
                      'Liberal':3,
                      'Very Liberal':4}
    cond_recode_dict = {'unpoliticized':0,
                       'left':1,
                       'right':2}
    state_recode_dict = dict(zip( np.unique(df['state']),np.arange(np.unique(df['state']).shape[0])))

    df['pol_recode'] = [pol_recode_dict[item] for item in df['politics']]
    df['social_correct'] = df['answer'] == df['social_response']
    df['asocial_correct'] = df['answer'] == df['asocial_response']
    df['p_recode'] = df['p'] == .98
    df['cond_recode'] = [cond_recode_dict[item] for item in df['condition']]
    df['state_recode'] = [state_recode_dict[item] for item in df['state']]

    return df


def clean_exp3(dat):
    data = dat[dat.index.values>1]

    #It just so happens that recently, scientists determined pluto is indeed larger. Let's fix that.
    data= data.rename(index=str,columns={'Maine_true_Conf_1':'Maine_True_Conf_1'})
    data.rename(columns={'Russia_True':'Russia_True_T','Russia_True_Conf_1':'Russia_True_Conf_1_T'})
    data.rename(columns={'Russia_False':'Russia_True','Russia_False_Conf_1':'Russia_True_Conf_1'})
    data.rename(columns={'Russia_True_T':'Russia_False','Russia_True_Conf_1_T':'Russia_False_Conf_1'})
    RICols = [item for item in data.columns if "RI_" in item]
    EICols = [item for item in data.columns if "EI_" in item]
    BSCols = [item for item in data.columns if "BS_" in item]
    DEMCols = [item for item in data.columns if "DEM_" in item]




    FalseAnswerCols = [item for item in data.columns if "False" in item and "Conf" not in item]
    TrueAnswerCols = [item for item in data.columns if "True" in item and "Conf" not in item]

    pol_recode_dict = {'Very Conservative':0, 'Conservative':1, 'Moderate':2,'Liberal':3,'Very Liberal':4}
    data['pol_recode'] = [pol_recode_dict[item] for item in data['DEM_2'].values]
    edu_recode_dict = {'Some High School':0,
                  'High School':1,
                  'Some College':2,
                  'College':3,
                  'Graduate Degree or Higher':4}
    data['edu_recode'] = [edu_recode_dict[item] for item in data.DEM_3]

    finished = []
    for idx in data.index.values:
        answers = (np.array([data[data.CintID==data['CintID'].loc[idx]][item].values[0] for item in RICols]))
        finishedAnswers = np.array([str(answer) != 'nan' for answer in answers])
        finished.append(np.all(finishedAnswers))
    data['finished'] = finished
    data = data[data.finished==True]

    RIDict = {'Definitely not true of myself 1':0,
         '2':1,
         'Not true or untrue 3':2,
         '4':3,
         'Definitely true of myself 5':4,
         np.nan:np.nan}
    RIDict_rev = {'Definitely not true of myself 1':4,
             '2':3,
             'Not true or untrue 3':2,
             '4':1,
             'Definitely true of myself 5':0,
             np.nan:np.nan}

    RISCore = np.array([[RIDict[item] for item in data['RI_1'].values],
    [RIDict[item] for item in data['RI_2'].values],
    [RIDict_rev[item] for item in data['RI_3'].values],
    [RIDict_rev[item] for item in data['RI_4'].values],
    [RIDict[item] for item in data['RI_5'].values],])

    RISCore = np.mean(RISCore,axis=0)
    data['RIS'] = RISCore


    data['HBS'] = np.sum(np.array([data['SS_Squash'].values == '15 points',
    data['GF1_Slot'].values == "1",
    data['DenNeglect'].values == 'small tray (10 marbles)',
    data['ProbMatch'].values == 'Strategy D: Predict red on all of the 60 rolls. ']),axis=0)


    data['WSS'] = np.sum(np.array([data['WS_1'].values == 'widen',
    data['WS_2'].values == "come",
    data['WS_3'].values == 'whim',
    data['WS_4'].values == 'reference',
    data['WS_5'].values == 'daring',]),axis=0)



    data['NUMS'] = np.sum(np.array([data['Num1'].values == "500",
    data['Num2'].values == '10',
    data['Num3'].values == '1 in 10']),axis=0)


    EIDict1 = {'Definitely not true of myself 1':0,
             '2':1,
             'Note true or untrue 3':2,
             '4':3,
             'Definitely true of myself 5':4,
             np.nan:np.nan}
    EIDict2 = {'Definitely not true of myself 1':0,
             '2':1,
             'Not true or untrue 3':2,
             '4':3,
             'Definitely true of myself 5':4,
             np.nan:np.nan}
    EIDict_rev = {'Definitely not true of myself 1':4,
             '2':3,
             'Not true or untrue 3':2,
             '4':1,
             'Definitely true of myself 5':0,
             np.nan:np.nan}

    EIScore = np.array([[EIDict1[item] for item in data['EI_1'].values],
    [EIDict2[item] for item in data['EI_2'].values],
    [EIDict_rev[item] for item in data['EI_3'].values],
    [EIDict_rev[item] for item in data['EI_4'].values],
    [EIDict_rev[item] for item in data['EI_5'].values],])

    EIScore = np.mean(EIScore,axis=0)
    data['EIS'] = EIScore

    BSRescore = {'Not at all profound':0, 'Somewhat profound':1,'Fairly profound':2, 'Very profound':3,'Definitely profound':4,np.nan:np.nan}
    BSScores= np.nanmean(np.array([[BSRescore[item] for item in data[BSQ].values] for BSQ in BSCols]),axis=0)
    data['BSS'] = BSScores


    BSRescore = {'Not at all profound':0, 'Somewhat profound':1,'Fairly profound':2, 'Very profound':3,'Definitely profound':4,np.nan:np.nan}

    dats = []
    for column in data.columns:
        if 'Conf' in column:
            tt = data[data[column.split('_')[0] + '_' +column.split('_')[1]].values.astype(str)!='nan']
            conf = tt[column].values.astype('int')
            answer = tt[column.split('_')[0] + '_' +column.split('_')[1]]

            correct = answer == column.split('_')[1]
            correct[answer.astype('str')=='nan'] = np.nan
            temp = {'conf':conf,
                        'answer':answer,
                        'question': [column.split('_')[0] + '_' +column.split('_')[1] for item in answer],
                        'correct':correct,
                        'correct_conf':[np.mean(conf[correct == True]) for item in answer],
                        'incorrect_conf':[np.mean(conf[correct == False]) for item in answer],

                        'difficulty':[np.mean(correct) for item in answer],
                        'pol_recode':tt.pol_recode,
                        'DEM_1': tt.DEM_1,
                        'DEM_2': tt.DEM_2,
                        'DEM_3': tt.DEM_3,
                        'DEM_4': tt.DEM_4,
                        'BSS' : tt.BSS,
                        'WSS' : tt.WSS,
                        'EIS' : tt.EIS,
                        'HBS' : tt.HBS,
                        'NUMS' : tt.NUMS,
                        'RIS' : tt.RIS,
                        'CintID' : tt.CintID,
                        'edu_recode': tt.edu_recode

                     }
            dats.append(temp)
    melted = pd.concat([pd.DataFrame(item).dropna() for item in dats])
    id_recode = dict(zip(np.unique(melted.CintID),np.arange(np.unique(melted.CintID).shape[0])+1))
    melted['id_recode'] = [id_recode[item] for item in melted.CintID]
    melted.answer = melted.answer.values == 'True'
    melted.conf = melted.conf /50 - 1.0
    melted['question_code'] = pd.Categorical(melted['question']).codes+1


    return data, melted

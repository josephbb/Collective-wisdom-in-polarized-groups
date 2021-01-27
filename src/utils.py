import numpy as np
import pandas as pd


def pickle_model(model, samples, model_location, samples_location, model_name):
    try:
        import cPickle as pickle
    except ModuleNotFoundError:
        import pickle
    with open(model_location+model_name+'_model.p', 'wb') as output:
        pickle.dump(model, output, pickle.HIGHEST_PROTOCOL)
    with open(samples_location+model_name+'_samples.p', 'wb') as output:
        pickle.dump(samples, output, pickle.HIGHEST_PROTOCOL)

def load_model(model_location, samples_location, model_name):
    try:
        import cPickle as pickle
    except ModuleNotFoundError:
        import pickle

    with open(model_location+model_name+'_model.p', 'rb') as input:
        model = pickle.load(input)
    with open(samples_location+model_name+'_samples.p', 'rb') as input:
        samples = pickle.load(input)


    return model, samples


def make_latex_table(samples, variables, q=[5.5,50.0,94.5]):
    dfs = []
    for var in variables:
        qs = np.percentile(samples[var], axis=0, q=q)
        item = pd.DataFrame({'variable': np.repeat(var, samples[var].shape[1]),
          'Mean':np.mean(samples[var], axis=0),
          'sd':np.std(samples[var], axis=0),
          str(q[0]) + '%':qs[0],
          str(q[1]) + '%':qs[1],
           str(q[2]) + '%':qs[2]})
        dfs.append(item)
    return pd.concat(dfs,sort=False).to_latex(index=False)

def save_latex_table(directory, name, table_string):
    with open(directory+'/'+name, "w") as text_file:
        text_file.write(table_string)

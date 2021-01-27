
data {
int<lower=0> n;
int<lower=0, upper=1> y[n];
int j;
int pol[j];
int who[n];
real confidence[n];
real socConfidence[n];
}

parameters {

real alpha_p[5];
real b_conf_p[5];
real b_socConf_p[5];

real alpha[j];
real b_conf[j];
real b_socConf[j];

}

model {
real theta[n];

for (pi in 1:5){
alpha_p[pi] ~ normal(0,2);
b_conf_p[pi] ~ normal(0,2);
b_socConf_p[pi] ~ normal(0,2);

}


for (idx in 1:j){

alpha[idx] ~ normal(alpha_p[pol[idx]],1);
b_conf[idx] ~ normal(b_conf_p[pol[idx]],1);
b_socConf[idx]~ normal(b_socConf_p[pol[idx]],1);

}

for (i in 1:n){
theta[i] = alpha[who[i]] + b_conf[who[i]] * confidence[i] +
                        b_socConf[who[i]]*socConfidence[i];

}

y ~ bernoulli_logit(theta);

}


generated quantities {
int yhat[n];
real theta_hat[n];

for (i in 1:n){
    theta_hat[i] = alpha[who[i]] + b_conf[who[i]] * confidence[i] +
                            b_socConf[who[i]]*socConfidence[i];
    yhat[i] = bernoulli_logit_rng(theta_hat[i]);
}


}

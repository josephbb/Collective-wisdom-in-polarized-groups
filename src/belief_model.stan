
data {
int n;
real<lower=0, upper=1> y[n];
int j;
int who[n];
int pol[j];
int<lower=0, upper=1> correct[n];
}

parameters {
real mu_a_pol[5];
real alpha_a_pol[5];
real gamma_a_pol[5];

real mu_b_pol[5];
real alpha_b_pol[5];
real gamma_b_pol[5];

real mu_a[j];
real gamma_a[j];
real alpha_a[j];
 

real mu_b[j];
real gamma_b[j];
real alpha_b[j];

real<lower=0.00001, upper=100> theta[5];

}

model {


real mu[n];
real alpha[n];
real gamma[n];

theta ~ normal(0,5);

for (pi in 1:5){
mu_a_pol[pi] ~ normal(0,1);
alpha_a_pol[pi] ~ normal(-1,1);
gamma_a_pol[pi] ~ normal(0,1);
}


for (pi in 1:5){
mu_b_pol[pi] ~  normal(0,.5);
alpha_b_pol[pi] ~  normal(0,.5);
gamma_b_pol[pi] ~  normal(0,.5);
}


for (jj in 1:j){
mu_a[jj] ~ normal(mu_a_pol[pol[jj]], .5);
alpha_a[jj] ~ normal(alpha_a_pol[pol[jj]], .5);
gamma_a[jj] ~ normal(gamma_a_pol[pol[jj]], .5);

mu_b[jj] ~ normal(mu_b_pol[pol[jj]], .5);
alpha_b[jj] ~ normal(alpha_b_pol[pol[jj]], .5);
gamma_b[jj] ~ normal(gamma_b_pol[pol[jj]], .5);


}


for (i in 1:n) {

 mu[i] = inv_logit(mu_a[who[i]] + mu_b[who[i]]*correct[i]);
 alpha[i] = inv_logit(alpha_a[who[i]] + alpha_b[who[i]]*correct[i]);
 gamma[i] = inv_logit(gamma_a[who[i]] + gamma_b[who[i]]*correct[i]);

 if(y[i] == 0) {
  target+= log(alpha[i]) + log1m(gamma[i]);
 } else if (y[i] == 1) {
    target += log(alpha[i]) + log(gamma[i]);
 } else {
   target += log1m(alpha[i]) + beta_proportion_lpdf(y[i] | mu[i], theta[pol[who[i]]]);
 }

}
}


generated quantities {

real yhat[n];
real muhat[n];
int aa[n];
int gg[n];

for (i in 1:n){

     muhat[i] = inv_logit(mu_a[who[i]] + mu_b[who[i]]*correct[i]);
     aa[i] = bernoulli_rng(inv_logit(alpha_a[who[i]] + alpha_b[who[i]]*correct[i]));
     gg[i] = bernoulli_rng(inv_logit(gamma_a[who[i]] + gamma_b[who[i]]*correct[i]));


    if (aa[i]==0){
        yhat[i] = beta_proportion_rng(muhat[i], theta[pol[who[i]]]);

    } else if (gg[i]==1){
        yhat[i] = 1;
    } else {
        yhat[i] = 0;
    }
    }

}

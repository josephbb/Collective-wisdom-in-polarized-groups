
data {
  int n;
  real x[n];
  int j;
  int who[n];
  int pol[j];
  int NStates;
  int state[n];
  int<lower=0, upper=1> correct[n];
}

parameters {
  real alpha[j];
  real beta[j];
  real accuracy[j];
  real alpha_p[5];
  real beta_p[5];
  real accuracy_p[5];
  vector[NStates-1] beta_state_raw;
}

transformed parameters {
  vector[NStates] beta_state = append_row(beta_state_raw, -sum(beta_state_raw));
}


model {
  real theta[n];
  real mu[n];

  for (p in 1:5){
    alpha_p[p] ~ normal(0,.4);
    beta_p[p] ~ normal(0,.4);
    accuracy_p[p] ~ normal(0,.4);
  }

  beta_state ~ normal(0,.4);



  for(jj in 1:j){
    alpha[jj] ~ normal(alpha_p[pol[jj]], .4);
    beta[jj] ~ normal(beta_p[pol[jj]], .4);
    accuracy[jj] ~ normal(accuracy_p[pol[jj]], .4);
  }

  for (i in 1:n) {
    theta[i] = alpha[who[i]] + beta[who[i]]*x[i] + beta_state[state[i]];
    mu[i] = accuracy[who[i]];
  }
  correct ~ bernoulli_logit(mu);
  correct ~ bernoulli_logit(theta);
}

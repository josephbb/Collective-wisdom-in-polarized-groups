
data {
  int<lower=0> N;              // num answers
  int<lower=1> K2;              // num predictors for individuals
  int<lower=1> K1;              // num predictors for questions
  int<lower=1> Q; //Number of Questions
  int<lower=1> S;              // num participants
  int<lower=1> L;              // group predictors
  int<lower=0, upper=1> y[N];                 // outcomes
  int<lower=1,upper=S> ss[N];  //  individual id
  int<lower=1,upper=S> qq[N];  //  individual id

  matrix[N, K2] x2;              // individual predictors
  matrix[N, K1] x1;              // individual predictors

  matrix[L, S] u;          // group predictors  
  real conf[N];
}

parameters {
  matrix[K2, S] m; //Parameter for subject-level effects
  matrix[K1, Q] qm; //Parameter for question-level effects
  
  real alpha; //Overall average accuracy
  real beta; //Overall relationship between accuracy and confidence

  cholesky_factor_corr[K2] L_Omega_s; //Participant effect covariance matrix
  cholesky_factor_corr[K1] L_Omega_q; //Question effect covariance matrix. 

  vector<lower=0,upper=pi()/2>[K2] tau_unif_s;  // prior scale
  vector<lower=0,upper=pi()/2>[K1] tau_unif_q;  // prior scale

  matrix[K2, L] gamma;                        // group coeffs
}

transformed parameters {
  vector<lower=0>[K2] tau_s = 2.5 * tan(tau_unif_s); 
  vector<lower=0>[K1] tau_q = 2.5 * tan(tau_unif_q);

  
  matrix[K2, S] beta_s = gamma * u + diag_pre_multiply(tau_s, L_Omega_s)*m; //Smuggling parameters out 
  matrix[K1, Q] beta_q =  diag_pre_multiply(tau_q, L_Omega_q)*qm; //Smuggling parameters out
}

model {
  real mu[N];
  alpha ~ normal(0,1);
  beta ~ normal(0,1);
  
  for(n in 1:N) {
    mu[n] =  alpha + beta * conf[n] + x1[n,]*beta_q[,qq[n]] + x2[n,]*beta_s[,ss[n]];
  }
 
  to_vector(m) ~ normal(0,1);
  to_vector(qm) ~ normal(0,1);

  to_vector(gamma) ~ normal(0,1);
  L_Omega_q ~ lkj_corr_cholesky(2);
  L_Omega_s ~ lkj_corr_cholesky(2);
  y ~ bernoulli_logit(mu);
}

generated quantities {
    int y_hat[N]; 
    int y_prior[N];
    real mu_prior[N];
    real mu_hat[N];
    real log_lik[N];
    matrix[K2, L] gamma_prior;
    matrix[K2, S] beta_s_prior;
    matrix[K1, Q] beta_q_prior;
    vector<lower=0>[K2] tau_s_prior;
    vector<lower=0>[K1] tau_q_prior;
    vector[K2] tau_unif_s_prior; 
    vector[K1] tau_unif_q_prior; 
    cholesky_factor_corr[K2] L_Omega_s_prior;
    cholesky_factor_corr[K1] L_Omega_q_prior;
    matrix[K2, S] m_prior;
    matrix[K1, Q] qm_prior;
    
 
    real alpha_prior = normal_rng(0,1);
    real beta_prior = normal_rng(0,1);
    for (k in 1:K2){
        for(l in 1:L){
        gamma_prior[k,l] = normal_rng(0,1);
        }
    }
    L_Omega_s_prior = lkj_corr_cholesky_rng(K2,2);
    L_Omega_q_prior = lkj_corr_cholesky_rng(K1,2);
    
    for(k in 1:K2){
        tau_unif_s_prior[k] = uniform_rng(0,pi()/2);
         for (s in 1:S) {
             m_prior[k,s] = normal_rng(0,1);
         }
    }
    for(k in 1:K1){
    tau_unif_q_prior[k] = uniform_rng(0,pi()/2);
        for(q in 1:Q){
        qm_prior[k,q] = normal_rng(0,1);
        }
    }
    

    tau_s_prior = 2.5 * tan(tau_unif_s_prior);
    tau_q_prior = 2.5* tan(tau_unif_q_prior);

    beta_s_prior = gamma_prior * u + diag_pre_multiply(tau_s_prior, L_Omega_s_prior)*m;
    beta_q_prior =  diag_pre_multiply(tau_q_prior, L_Omega_q_prior)*qm;

        for(n in 1:N){
            mu_hat[n] =  alpha + beta * conf[n] + x1[n,]*beta_q[,qq[n]] + x2[n,]*beta_s[,ss[n]];
            y_hat[n] = bernoulli_logit_rng(mu_hat[n]);
            log_lik[n] = bernoulli_logit_lpmf(y[n] | mu_hat[n]);
        }

        for(n in 1:N){
        mu_prior[n] = alpha_prior + 
                        beta_prior * conf[n] + 
                        x1[n,]*beta_q_prior[,qq[n]] + 
                        x2[n,]*beta_s_prior[,ss[n]];
        y_prior[n] = bernoulli_logit_rng(mu_prior[n]);
        }


    }


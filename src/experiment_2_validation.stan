data {
int N;
int cond[N];
int y[N];
int count[N];
}
parameters{
real alpha[6];
}


model {
real mu[N];

alpha ~ normal(0,.2);

for(n in 1:N){
mu[n] = inv_logit(alpha[cond[n]]);
y[n] ~ binomial(count[n],mu[n]);
}


}

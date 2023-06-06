data{
  int<lower=1> N;
  real<lower=0,upper=100> pi[N];
  real<lower=0,upper=100> u[N];
}

parameters{
  real L[N]; //lambda*phi
  real<lower=0> un; //natural unemployment rate
  real<lower=-20,upper=20> beta[N];
  real uc[3];
  real cL[3];
  real b[3];
  real<lower=0> sb;
  real<lower=0> su;
  real<lower=0> sL;
  real<lower=0> s;
}

model{
  real p[N-1];

  for(i in 3:(N)){
             u[i]~normal(uc[1]+uc[2]*u[i-1]+uc[3]*u[i-2],su);
             L[i]~normal(cL[1]+cL[2]*L[i-1]+cL[3]*L[i-2],sL);
             beta[i]~normal(b[1]+b[2]*beta[i-1]+b[3]*beta[i-2],sb);
  }

  for(i in 1:(N-1))
             p[i]<-beta[i]*pi[i+1]-L[i]*(u[i]-un);

  for(i in 1:(N-1))
             pi[i]~normal(p[i],s);

  su~inv_gamma(0.0001,0.0001);
  s~inv_gamma(0.0001,0.0001);
  sL~inv_gamma(0.0001,0.0001);
  sb~inv_gamma(0.0001,0.0001);

}
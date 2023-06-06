data{
  int<lower=1> N;
  real<lower=0,upper=100> pi[N];
  real<lower=0,upper=100> c[N];
  real<lower=0,upper=100> u[N];
}

parameters{
  real L[N]; //lambda*phi
  real<lower=0> un; //natural unemployment rate
  real alpha;
  real<lower=-20,upper=20> beta[N];
  real gamma[N];

  real uu[N];
  real cc[N];

  real uc[3];
  real cL[3];
  real b[3];
  real g[3];

  real<lower=0> su;
  real<lower=0> sL;
  real<lower=0> sb;
  real<lower=0> sg;
  real<lower=0> s;

  real<lower=0> soc;
  real<lower=0> sou;

}

model{
  real p[N];

  for(i in 3:(N)){
  u[i]~normal(uc[1]+uc[2]*u[i-1]+uc[3]*u[i-2],su);
  L[i]~normal(cL[1]+cL[2]*L[i-1]+cL[3]*L[i-2],sL);

  beta[i]~normal(b[1]+b[2]*beta[i-1]+b[3]*beta[i-2],sb);
  gamma[i]~normal(g[1]+g[2]*gamma[i-1]+g[3]*gamma[i-2],sg);
  }
  for(i in 1:(N)){
  u[i]~normal(uu[i],sou);
  c[i]~normal(cc[i],soc);
  }

  for(i in 2:(N-1))
  p[i]<-alpha+gamma[i]*cc[i-1]+beta[i-1]*pi[i+1]-L[i]*(uu[i]-un);

  for(i in 2:(N-1))
  pi[i]~normal(p[i]-gamma[i]*cc[i-1],s);
}
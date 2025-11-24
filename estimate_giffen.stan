data {
  int<lower=1> N;         // 観測数
  int<lower=1> I;         // 品目数
  int<lower=1,upper=I> item[N];
  vector[N] log_q;        // log quantity
  vector[N] log_Y;        // log income
  vector[N] log_P;        // log price index
}

parameters {
  real eta_bar;           // 所得弾力性の平均
  real<lower=0> tau_eta;  // 所得弾力性の標準偏差
  vector[I] eta_tilde;    // 標準化された所得弾力性

  real gamma_bar;         // 価格弾力性の平均
  real<lower=0> tau_gamma;// 価格弾力性の標準偏差
  vector[I] gamma_tilde;  // 標準化された価格弾力性

  vector[I] a;            // 品目別定数項
  real<lower=0> sigma;    // 観測誤差
}

transformed parameters {
  vector[I] eta;
  vector[I] gamma;

  eta = eta_bar + tau_eta * eta_tilde;
  gamma = gamma_bar + tau_gamma * gamma_tilde;
}

model {
  // 事前分布
  eta_tilde   ~ normal(0, 1);
  gamma_tilde ~ normal(0, 1);

  eta_bar   ~ normal(0, 1);     // 所得弾力性は 0 近傍に弱情報
  tau_eta   ~ cauchy(0, 1);

  gamma_bar ~ normal(0, 1);     // 価格弾力性も 0 近傍
  tau_gamma ~ cauchy(0, 1);

  a         ~ normal(0, 5);
  sigma     ~ cauchy(0, 1);

  // 尤度
  for (n in 1:N) {
    log_q[n] ~ normal(a[item[n]]
                      + eta[item[n]]   * log_Y[n]
                      + gamma[item[n]] * log_P[n],
                      sigma);
  }
}

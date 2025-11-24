// nkpc_panel.stan
data {
  int<lower=1> N;            // 観測数（item×time の総数）
  int<lower=1> J;            // 品目数
  int<lower=1,upper=J> item[N]; // 観測ごとの品目インデックス
  vector[N] pi;              // インフレ率
  vector[N] pi_fwd;          // 期待インフレ proxy (lead など)
  vector[N] x;               // マクロ・スラック（全品目共通）
}

parameters {
  real beta;                 // NKPC の期待インフレ係数
  real kappa_bar;            // 平均スロープ
  real<lower=0> tau_kappa;   // スロープの分散
  vector[J] kappa_tilde;     // スロープの標準化 SH
  real alpha_bar;            // 平均定数項
  real<lower=0> tau_alpha;   // 定数項の分散
  vector[J] alpha_tilde;     // 定数項の標準化
  real<lower=0> sigma;       // 誤差の分散
}

transformed parameters {
  vector[J] alpha;
  vector[J] kappa;
  alpha = alpha_bar + tau_alpha * alpha_tilde;
  kappa = kappa_bar + tau_kappa * kappa_tilde;
}

model {
  // 階層事前
  alpha_tilde ~ normal(0, 1);
  kappa_tilde ~ normal(0, 1);

  beta ~ normal(1, 0.5);        // 例：事前に1 近傍
  kappa_bar ~ normal(0, 0.5);   // スロープは小さめ
  tau_alpha ~ cauchy(0, 0.5);
  tau_kappa ~ cauchy(0, 0.5);
  sigma ~ cauchy(0, 0.5);

  // 観測方程式
  for (n in 1:N) {
    pi[n] ~ normal(alpha[item[n]] + beta * pi_fwd[n] + kappa[item[n]] * x[n], sigma);
  }
}

"""
CPI (price_index) + 家計調査（quantity, income）から
PyStan を用いてギッフェン財候補を推定するスクリプト。

前提となる tidy CSV:
  - cpi_tidy.csv:  year,item,price_index
  - kakei_tidy.csv: year,item,quantity,income

使い方:
  python estimate_giffen.py \
      --cpi cpi_tidy.csv \
      --kakei kakei_tidy.csv \
      --out giffen_results.csv
"""
import argparse
import numpy as np
import pandas as pd
import pystan

def build_panel(cpi_path: str, kakei_path: str) -> pd.DataFrame:
  """
  cpi_tidy.csv と kakei_tidy.csv を読み込み、
  year & item で内部結合してパネルデータを作る。
  想定カラム:
    cpi_tidy.csv:    year, item, price_index
    kakei_tidy.csv:  year, item, quantity, income
  """
  cpi = pd.read_csv(cpi_path)
  kakei = pd.read_csv(kakei_path)

  # 型の統一
  cpi["year"] = cpi["year"].astype(int)
  kakei["year"] = kakei["year"].astype(int)

  # 内部結合
  df = pd.merge(
      kakei,
      cpi,
      on=["year", "item"],
      how="inner",
      validate="many_to_many",
  )

  # 負やゼロを除外（log をとるため）
  df = df[(df["quantity"] > 0) & (df["income"] > 0) & (df["price_index"] > 0)]

  # ログを作る
  df["log_q"] = np.log(df["quantity"])
  df["log_Y"] = np.log(df["income"])
  df["log_P"] = np.log(df["price_index"])

  # 品目 ID の付与（Stan 用に 1 からの整数）
  item_codes = sorted(df["item"].unique())
  item_to_id = {item: i + 1 for i, item in enumerate(item_codes)}
  df["item_id"] = df["item"].map(item_to_id).astype(int)

  return df, item_codes


def build_stan_data(panel: pd.DataFrame):
  """
  パネル DataFrame から Stan の data を構築。
  panel: columns = year, item, item_id, log_q, log_Y, log_P, ...
  """
  panel = panel.sort_values(["item_id", "year"])
  N = len(panel)
  I = panel["item_id"].nunique()

  stan_data = {
      "N": int(N),
      "I": int(I),
      "item": panel["item_id"].astype(int).values,
      "log_q": panel["log_q"].values.astype(float),
      "log_Y": panel["log_Y"].values.astype(float),
      "log_P": panel["log_P"].values.astype(float),
  }
  return stan_data


def run_stan(modelname,stan_data, iter=4000, chains=4, seed=1234):
  """
  Stan モデルをコンパイルしてサンプリングを実行。
  """
  with open(modelname) as f:
      stan_model=f.read()

  sm = pystan.StanModel(model_code=stan_model)
  fit = sm.sampling(
      data=stan_data,
      iter=iter,
      chains=chains,
      seed=seed,
      n_jobs=-1
  )
  return fit


def summarize_giffen(fit, item_codes):
  """
  Stan の sampling 結果から、
  各品目の eta, gamma の事後分布を取り出し、
  ギッフェン候補度を計算して DataFrame にまとめる。

  Giffen 候補の指標:
    - p_eta_neg = P(eta < 0)
    - p_gamma_pos = P(gamma > 0)
    - p_giffen = P(eta < 0 & gamma > 0)
  """
  samples = fit.extract(permuted=True)
  eta = samples["eta"]      # shape: (S, I)
  gamma = samples["gamma"]  # shape: (S, I)

  S, I = eta.shape

  records = []
  for i in range(I):
    eta_i = eta[:, i]
    gamma_i = gamma[:, i]

    p_eta_neg = np.mean(eta_i < 0.0)
    p_gamma_pos = np.mean(gamma_i > 0.0)
    p_giffen = np.mean((eta_i < 0.0) & (gamma_i > 0.0))

    # 事後平均と HPDI 的な区間もつけておく
    eta_mean = eta_i.mean()
    gamma_mean = gamma_i.mean()
    eta_ci = np.quantile(eta_i, [0.025, 0.975])
    gamma_ci = np.quantile(gamma_i, [0.025, 0.975])

    records.append({
        "item": item_codes[i],
        "eta_mean": eta_mean,
        "eta_2.5%": eta_ci[0],
        "eta_97.5%": eta_ci[1],
        "gamma_mean": gamma_mean,
        "gamma_2.5%": gamma_ci[0],
        "gamma_97.5%": gamma_ci[1],
        "p_eta_neg": p_eta_neg,
        "p_gamma_pos": p_gamma_pos,
        "p_giffen": p_giffen,
    })

  result_df = pd.DataFrame.from_records(records)
  result_df = result_df.sort_values("p_giffen", ascending=False)
  return result_df


def main():
  parser = argparse.ArgumentParser(
      description="CPI + 家計調査からギッフェン財候補をベイズ推定するスクリプト"
  )
  parser.add_argument("--cpi", required=True, help="cpi_tidy.csv のパス (year,item,price_index)")
  parser.add_argument("--kakei", required=True, help="kakei_tidy.csv のパス (year,item,quantity,income)")
  parser.add_argument("--out", required=True, help="結果を保存する CSV パス")
  parser.add_argument("--iter", type=int, default=4000, help="Stan の反復数")
  parser.add_argument("--chains", type=int, default=4, help="Stan のチェーン数")
  parser.add_argument("--modelname", type=str, default="estimate_giffen.stan", help=".stan model name")
  args = parser.parse_args()

  print(">>> パネルデータを構築中 ...")
  panel, item_codes = build_panel(args.cpi, args.kakei)

  print("観測数 N =", len(panel), "品目数 I =", len(item_codes))
  print("対象品目:", item_codes)

  stan_data = build_stan_data(panel)

  print(">>> Stan によるサンプリングを開始 ...")
  fit = run_stan(modelname,stan_data, iter=args.iter, chains=args.chains)
  print(fit)

  print(">>> ギッフェン候補度を集計 ...")
  result_df = summarize_giffen(fit, item_codes)

  result_df.to_csv(args.out, index=False, encoding="utf-8-sig")
  print("結果を保存しました:", args.out)
  print("上位数行:")
  print(result_df.head(20))


if __name__ == "__main__":
  main()

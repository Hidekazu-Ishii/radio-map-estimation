# 実験：<FFNN パスロスモデルのチューニング>

- GitHub Issue Number: `#<Issue No.17>`
- Config: `configs/tuning/ffnn_model_resarch.yaml` 
- Command(tuning):
  - `uv run scripts/tuning/ffnn_pathloss.py ffnn configs/plateau.yaml configs/sionna.yaml configs/tuning/ffnn_model_search.yaml`
  - `uv run scripts/tuning/ffnn_pathloss.py ffnn_los configs/plateau.yaml configs/sionna.yaml configs/tuning/ffnn_model_search.yaml`
- Command(analize):
  - `uv run scripts/tuning/analyze_ffnn_pathloss.py ffnn tune_v1`
  - `uv run scripts/tuning/analyze_ffnn_pathloss.py ffnn_los tune_v1`
- Outputs(tuinig): `outputs/tuning/{city_dir}/{mesh_code}/{freq_ghz}/{search_dir_name}/{run_id}/{param_hash}/`
- Outputs(analuze): `outputs/tuning_analysis/{model_name}/`
- Notes: GPU=..., dataset version=...


## 問い・仮説

- エリア-周波数ごとにチューニングを行い，シャドウイング成分が抽出されることを確かめたい

## 実験内容

- 非test領域の中で，train=512, test=100 をランダムに選び，グリッドサーチを行った．（1条件につき10回ランダムにtrain, testを決める）

## わかったこと

## 次に検証すべきこと (あれば書く)

- 次にどんな実験をすべきかを書く

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

- FFNNLOS モデルは性能が良すぎて，シャドウイング成分の抽出がうまくできない（値にばらつきがない）
→実測値を使わない本実験では使わない方がいい？
- パスロスモデル（FFNNLos+フレネル楕円建物遮蔽特徴量）が高精度であるため、残差（シャドウイング成分）にはGudmundsonカーネル(等方的・定常)で検出できるような空間相関構造がほとんど残っておらず，d_corが極端に短く推定されている．これはサンプリング密度不足によるものではなく（train間隔を狭めても再現される），パスロスモデルが大域的な遮蔽構造を実質的に説明しきっていることに起因すると考えられる．
→Deep Kernel（非定常・非等方カーネル）がこの残された局所構造をどこまで拾えるかの検証に進む

## 次に検証すべきこと (あれば書く)

- 次にどんな実験をすべきかを書く

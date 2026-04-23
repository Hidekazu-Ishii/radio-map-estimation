# Devcontainerを使うpython用のテンプレート

このリポジトリは，研究室の **pythonの機械学習プロジェクト用テンプレート**です．  
このテンプレートでは，Devcontainerを使って環境構築をします．またこのテンプレートであらかじめ設定されているDevcontainerはGPUを使う前提で設定がなされているので注意してください．

DevContainerの使い方は以下の資料を参考にしてください:  
[Dev Containerによる開発環境の構築](https://www.docswell.com/s/2625216247/KEYDXG-2025-11-02-173012)



## 使い方

まず，GPUが乗ったマシンにはdockerやNVIDIA Container Toolkit，nvidiaのドライバがインストールされていることが前提条件です．
もしインストールされていない場合はインストールしておきましょう．

1. 新しくリポジトリを作るタイミングで，GitHub の **`Start with a template`** からこのテンプレートを選んでリポジトリを作成する。
2. 作成したリポジトリを clone する。
3. `pyproject.toml` の `[project]` セクションの `name`, `description`, `authors` などをプロジェクト用に変更する。
4. `src/lab_template_python` の `lab_template_python` を適切なパッケージ名に変更する。
5. （Python のバージョンを変更する場合）
   - `pyproject.toml` の `[project]` の `requires-python` を変更する  
     （例: Python 3.10 を使いたいなら `">=3.10"` に変更）
   - `pyproject.toml` の `[tool.ruff]` の `target-version` を変更する  
     （例: `py310`）
   - `.python-version` の中身を使用するバージョンに変更する  
     （例: `3.10`）
6. `.devcontainer`のフォルダの中の`Dockerfile`を編集して，自身が使いたい`cuda`のバージョンを指定します．`FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`のところで，動かしたいプロジェクトや自身の計算機の環境に応じて，必要なバージョンのcuda imageを設定します．
[nvidia-cudaのサイト](https://hub.docker.com/r/nvidia/cuda/tags)からimageを探して，書き換えてください．
(例) 12.8.0-cudnn-devel-ubuntu22.04を使いたい場合は，`FROM nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04`と書き換えます．
7. VSCodeの左下の`><`をクリックし，`Reopen in Container`をクリックして開発環境を構築します．
初めてコンテナを起動する時には，数分時間がかかるかと思います．
無事開発環境が立ち上がったら，ターミナルで`nvidia-smi`を実行して，`cuda`がインストールされているか確認しましょう．


## (Optional) PyTorchを入れる

pytorchを使いたい場合は，[uvの公式サイト](https://docs.astral.sh/uv/guides/integration/pytorch/#installing-pytorch) を参考にしつつ，`pytorch`を入れましょう．
何も考えずに`uv add torch`をしてしまうと，`cuda`のバージョンと`pytorch`のバージョンが合致せず，正しく実行できない可能性があります．
上記サイトのやり方でuvを使ってpytorchを入れましょう！


## コンテナの運用方法

開発をしているとpython以外のライブラリが必要になる場合があると思います．その時は，忘れずに`Dockerfile`でそのライブラリをインストールするコマンドを追加しておくと，実行環境の再現性が保たれます．

### 実際にあったケース

`opencv`というpythonパッケージを`uv add`してimportしようとすると
```
ImportError: libGL.so.1: cannot open shared object file: No such file or directory
```

というエラーが出る．これは必要なpython外のライブラリが入っていないと発生するエラーで，`libgl1`と`libglib2.0-0`をインストールするとエラーが発生しなくなる．このとき，忘れずに以下のようにDockerfileに追記して，自動でインストールできるようにしておくと良いです．

追記前:
```
RUN apt-get update && apt-get install -y \
    tmux \
    && rm -rf /var/lib/apt/lists/*
```

追記後:
```
RUN apt-get update && apt-get install -y \
    tmux \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
```

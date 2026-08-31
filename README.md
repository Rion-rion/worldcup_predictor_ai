# ⚽ World Cup Predictor AI

サッカー国際Aマッチの過去データ、Eloレーティング、選手データ、機械学習、Poisson分布、モンテカルロシミュレーションを組み合わせて、代表戦の勝敗確率やグループリーグ突破率を予測するPythonプロジェクトです。

Google Colab上で作成した予測AIをリファクタリングし、現在は **Visual Studio Code / Git / GitHub / Docker** を利用してローカル環境でも再現・実行できる構成にしています。

## 🎯 プロジェクトの目的

サッカーの試合結果を単純な勝敗予測だけでなく、複数の観点から分析することを目的としています。

- 過去の代表戦成績
- Eloレーティング
- 直近5試合・10試合の成績
- 選手能力
- 市場価値
- 得点・アシスト
- 出場時間
- Poisson分布による得点予測
- RandomForestによる勝敗予測
- モンテカルロシミュレーションによる大会予測

現在は、日本・オランダ・スウェーデン・チュニジアの4カ国を対象にグループリーグをシミュレーションします。

---

## 🚀 主な機能

### 1. データ取得・前処理

国際Aマッチの過去データとKaggleの選手データを取得・整形します。

主な処理：

- 国際Aマッチデータの読み込み
- KaggleHubによる選手データ取得
- 欠損値処理
- 選手データの結合
- 選手スコア算出
- 代表選手データの読み込み
- 国別チーム力の算出

---

### 2. AIによる勝敗予測

`RandomForestClassifier` を利用して、

- ホーム勝利
- 引き分け
- アウェイ勝利

の3クラス分類を行います。

主な特徴量：

- 直近5試合の勝率
- 直近10試合の勝率
- 平均得点
- 平均失点
- チーム間の成績差
- Eloレーティング
- Elo差
- Eloによる期待勝率
- 中立地フラグ

データリークを避けるため、試合日時順に並べ、

- 古い80%：学習データ
- 新しい20%：テストデータ

として評価しています。

実行時には以下を表示します。

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

---

### 3. Poisson分布によるスコア予測

選手データからチーム力を計算し、両チームの期待得点（xG）を算出します。

その期待得点を利用してPoisson分布から得点を生成し、多数回の試合シミュレーションを行います。

出力例：

- 日本勝利確率
- 引き分け確率
- 相手勝利確率
- 日本の期待得点
- 相手の期待得点
- 最頻スコア
- スコア分布

---

### 4. グループリーグ全6試合予測

対象グループ：

- 🇯🇵 Japan
- 🇳🇱 Netherlands
- 🇸🇪 Sweden
- 🇹🇳 Tunisia

以下の全6試合を予測します。

1. 日本 vs オランダ
2. 日本 vs スウェーデン
3. 日本 vs チュニジア
4. オランダ vs スウェーデン
5. オランダ vs チュニジア
6. スウェーデン vs チュニジア

各試合について、

- Poisson勝敗確率
- RandomForest勝敗確率
- 期待得点
- 最頻スコア

をまとめて表示します。

---

### 5. 日本代表3試合の勝敗確率

グループリーグ全6試合から日本代表の3試合を抽出し、

- 日本勝利
- 引き分け
- 日本敗北

の確率を比較します。

グラフによって対戦相手ごとの勝敗確率を視覚化します。

---

### 6. グループリーグ突破率

4カ国のグループリーグ全体を10,000回シミュレーションし、各国について以下を算出します。

- 1位確率
- 2位確率
- グループ突破率
- 平均勝ち点
- 平均得失点差
- 平均得点

さらに日本代表について、

- 1位通過率
- 2位通過率
- グループ突破率
- 平均勝ち点

を最終予測として表示します。

---

## 🔄 処理フロー

```text
データ取得・前処理
        ↓
Eloレーティング計算
        ↓
直近5試合・10試合の特徴量作成
        ↓
RandomForest学習・評価
        ↓
日本 vs オランダ 試合ログ生成
        ↓
グループリーグ全6試合予測
        ↓
日本代表3試合の勝敗確率
        ↓
モンテカルロシミュレーション
        ↓
グループリーグ突破率算出
        ↓
グラフ表示
```

---

## 📊 主な出力

### AIモデル評価

```text
AIモデル精度
分類レポート
混同行列
```

### グループリーグ全6試合

```text
日本 vs オランダ
日本 vs スウェーデン
日本 vs チュニジア
オランダ vs スウェーデン
オランダ vs チュニジア
スウェーデン vs チュニジア
```

### 日本代表

```text
オランダ戦      勝利 / 引き分け / 敗北
スウェーデン戦  勝利 / 引き分け / 敗北
チュニジア戦    勝利 / 引き分け / 敗北
```

### グループリーグ

```text
1位確率
2位確率
突破率
平均勝ち点
平均得失点差
平均得点
```

---

## 🛠 使用技術

### Language

- Python 3.13

### Data Analysis

- pandas
- NumPy
- openpyxl

### Machine Learning

- scikit-learn
- RandomForestClassifier

### Visualization

- matplotlib

### Data Source / Acquisition

- KaggleHub

### Development / Infrastructure

- Visual Studio Code
- Git
- GitHub
- Docker
- WSL2

---

## 🧠 使用アルゴリズム

### Random Forest

過去の代表戦データから作成した特徴量を利用して、試合結果を3クラス分類します。

### Elo Rating

代表チームの過去の試合結果から、チームの相対的な強さを数値化します。

### Poisson Distribution

期待得点をもとに各チームの得点を確率的に生成します。

### Monte Carlo Simulation

多数回の試合・大会シミュレーションを実施し、勝率やグループ突破率を算出します。

---

## 📁 ディレクトリ構成

```text
worldcup_predictor_ai/
│
├── data/
│   ├── README.md
│   └── world_cup_ai_4countries_japanese.xlsx
│
├── ai_model.py
├── data_prepare.py
├── match_simulation.py
├── japan_group_simulation.py
├── main.py
│
├── requirements.txt
├── Dockerfile
├── .dockerignore
├── .gitignore
├── README.md
│
└── worldcup_predictor_ai_clean_ipynb2.ipynb
```

---

## 📄 各ファイルの役割

### `data_prepare.py`

データ取得・前処理を担当します。

主な処理：

- 国際Aマッチデータ取得
- Kaggle選手データ取得
- データ結合
- 欠損値処理
- 選手特徴量作成
- 選手スコア算出
- 代表メンバー読み込み

---

### `ai_model.py`

機械学習モデルを担当します。

主な処理：

- Eloレーティング計算
- 直近5試合・10試合特徴量作成
- RandomForestClassifier学習
- 時系列Train/Test分割
- 勝敗確率予測
- Accuracy算出
- Classification Report
- Confusion Matrix

---

### `match_simulation.py`

1試合およびグループリーグのシミュレーションを担当します。

主な処理：

- 代表メンバー抽出
- チーム力算出
- 期待得点算出
- Poissonによる得点生成
- スコア分布算出
- 勝敗確率算出
- 試合ログ生成
- グループ順位計算

---

### `japan_group_simulation.py`

4カ国のグループリーグ予測を担当します。

主な処理：

- グループ全6試合予測
- 全6試合結果の一覧表示
- 日本代表3試合の抽出
- 日本の勝敗確率グラフ
- 4カ国の突破率計算
- 日本の1位・2位・突破率表示

---

### `main.py`

プロジェクト全体のエントリーポイントです。

以下の処理を順番に実行します。

```text
データ準備
↓
AIモデル学習
↓
試合ログ生成
↓
グループリーグシミュレーション
↓
結果表示
↓
グラフ表示
```

---

## 📊 使用データ

### 国際Aマッチデータ

過去の代表戦結果を使用します。

主な項目：

- 試合日
- ホームチーム
- アウェイチーム
- ホーム得点
- アウェイ得点
- 中立地
- 勝敗結果

---

### Kaggle選手データ

KaggleHubを利用して以下のデータセットを取得します。

```text
davidcariboo/player-scores
```

主に利用するデータ：

```text
players.csv
appearances.csv
player_valuations.csv
```

ローカルに必要なCSVが存在しない場合、KaggleHubを利用してデータを取得します。

そのため、初回実行時などはインターネット接続が必要になる場合があります。

---

### 4カ国代表データ

以下のExcelファイルを使用します。

```text
data/world_cup_ai_4countries_japanese.xlsx
```

対象国：

- 日本
- オランダ
- スウェーデン
- チュニジア

---

## 🚀 ローカル環境での実行

### 1. Repositoryをclone

```bash
git clone https://github.com/Rion-rion/worldcup_predictor_ai.git
cd worldcup_predictor_ai
```

### 2. 仮想環境を作成

```bash
python -m venv .venv
```

### Windows PowerShell

必要に応じて：

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

仮想環境を有効化：

```powershell
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
source .venv/bin/activate
```

### 3. ライブラリをインストール

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. 実行

```bash
python main.py
```

デフォルトでは10,000回のシミュレーションを実行し、最後にグラフを表示します。

---

## ⚙️ オプション

### シミュレーション回数を変更

例：50,000回

```bash
python main.py --simulations 50000
```

### グラフを表示しない

```bash
python main.py --no-graphs
```

---

## 🐳 Docker

Dockerを利用することで、ローカルのPython環境に依存しない実行環境を構築できます。

### Docker Image Build

```bash
docker build -t worldcup-predictor-ai .
```

### Container Run

```bash
docker run --rm worldcup-predictor-ai
```

Docker環境では `MPLBACKEND=Agg` を使用しているため、GUIによるMatplotlibウィンドウ表示は行いません。

Docker Desktop / WSL2環境で動作確認しています。

---

## 📦 requirements.txt

主要ライブラリ：

```text
pandas
numpy
matplotlib
scikit-learn
openpyxl
kagglehub
```

---

## 🔄 Google Colab版からの変更

元々Google Colab上で作成していたコードを、ローカル開発向けにリファクタリングしました。

主な変更：

- Colab依存パスの削除
- Pythonコードを機能別に分割
- VS Codeで実行可能な構成へ変更
- `requirements.txt` による依存関係管理
- Python 3.13対応
- Git / GitHubによるバージョン管理
- `.gitignore` の追加
- Dockerfileの追加
- `.dockerignore` の追加
- Windows / Linux向け日本語フォント対応
- Docker / WSL2環境での動作確認
- グループリーグ全6試合予測
- 日本代表3試合の勝敗確率表示
- グループ突破率シミュレーション
- CLIオプション追加

---

## ⚠️ 現在の制約

このプロジェクトは予測・分析を目的とした個人開発プロジェクトです。

予測結果は以下の情報に依存します。

- 使用した過去試合データ
- 選手データ取得時点
- 市場価値
- モデルの特徴量
- シミュレーション条件

そのため、実際の試合結果を保証するものではありません。

また、現在のグループ順位判定では、勝ち点・得失点差・得点が同一の場合の最終順位決定を簡略化しています。

---

## 🔮 今後の改善

- 4カ国から48カ国への拡張
- FIFAランキング追加
- 選手コンディション追加
- 怪我情報の反映
- スタメン・フォーメーション情報の反映
- ホーム・アウェイ補正改善
- RandomForest以外のモデル比較
- XGBoost / LightGBMとの比較
- ベースラインモデルとの精度比較
- Balanced Accuracy / Macro F1などの評価追加
- グラフのPNG自動保存
- READMEへの予測結果画像掲載
- Kaggleデータのローカルキャッシュ強化
- GitHub Actionsによる自動テスト
- Webアプリ化
- REST API化
- 2026年W杯48カ国対応

---

## 📌 Status

- ✅ Google Colabからローカル環境へ移行
- ✅ VS Codeで実行可能
- ✅ Python 3.13対応
- ✅ Git / GitHub管理
- ✅ RandomForestによる勝敗予測
- ✅ Eloレーティング実装
- ✅ Poisson得点シミュレーション
- ✅ モンテカルロシミュレーション
- ✅ グループ全6試合予測
- ✅ 日本代表3試合の勝敗確率表示
- ✅ グループ突破率算出
- ✅ Matplotlibグラフ表示
- ✅ Dockerfile作成
- ✅ Dockerコンテナ動作確認

---

## 👤 Author

Rion-rion

Python / Data Analysis / Machine Learning Portfolio Project
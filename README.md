# ⚽ World Cup Predictor AI

代表戦データ、Eloレーティング、選手データ、機械学習、
モンテカルロシミュレーションを利用して、
サッカー代表戦の勝敗・スコア分布・グループリーグ突破確率を予測する
Pythonプロジェクトです。

元々Google Colab上で実行していたコードをリファクタリングし、
現在は **Visual Studio Code / GitHub / Docker** を利用して
ローカル環境でも再現・実行できる構成にしています。

---

## 📌 主な機能

- 国際Aマッチの過去データ取得・前処理
- KaggleHubを利用した選手データ取得
- 選手市場価値・出場時間・得点・アシストを利用した選手スコア算出
- 国別チーム力の算出
- Eloレーティングの計算
- 直近5試合・10試合の特徴量作成
- RandomForestClassifierによる勝敗予測
- ポアソン分布を利用した得点シミュレーション
- 10,000回のモンテカルロシミュレーション
- スコア分布の算出
- グループリーグ順位・突破確率の算出
- 選手名を利用した試合ログ生成
- Matplotlibによるグラフ表示

---

## 🤖 予測に使用する主な要素

試合予測では、主に以下の情報を利用します。

- 過去の代表戦結果
- Eloレーティング
- 直近の勝敗傾向
- 平均得点
- 平均失点
- 選手市場価値
- 出場時間
- 得点
- アシスト
- チーム総合力

これらの特徴量を組み合わせ、
機械学習モデルと確率シミュレーションによって試合結果を予測します。

---

## 🛠 使用技術

### 言語

- Python

### 開発環境

- Visual Studio Code
- Git
- GitHub
- Docker
- WSL2

### 主なライブラリ

- pandas
- NumPy
- matplotlib
- scikit-learn
- openpyxl
- kagglehub

### 主なアルゴリズム・手法

- RandomForestClassifier
- Eloレーティング
- ポアソン分布
- モンテカルロシミュレーション

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
📄 各ファイルの役割
data_prepare.py

データ取得・前処理を担当します。

主な処理：

国際Aマッチデータ取得
Kaggle選手データ取得
欠損値処理
選手特徴量作成
選手スコア算出
4カ国代表Excel読み込み
ai_model.py

機械学習モデルの学習・予測を担当します。

主な処理：

Eloレーティング計算
直近試合データから特徴量作成
RandomForestClassifierの学習
勝敗確率の予測
モデル精度評価
match_simulation.py

1試合単位のシミュレーションを担当します。

主な処理：

チーム力算出
期待得点（xG）算出
ポアソン分布による得点生成
10,000回シミュレーション
勝率・引き分け率算出
スコア分布表示
試合ログ生成
japan_group_simulation.py

日本代表が所属するグループリーグ全体をシミュレーションします。

対象国：

🇯🇵 Japan
🇳🇱 Netherlands
🇸🇪 Sweden
🇹🇳 Tunisia

主な出力：

1位確率
2位以内突破確率
平均勝ち点
平均得失点差
main.py

各処理を順番に実行するエントリーポイントです。

データ準備
    ↓
AIモデル学習
    ↓
試合シミュレーション
    ↓
グループリーグシミュレーション
📊 使用データ
国際Aマッチデータ

過去の代表戦結果を取得して利用します。

主な項目：

試合日
ホームチーム
アウェイチーム
得点
勝敗結果
Kaggle選手データ

KaggleHubの以下のデータセットを利用します。

davidcariboo/player-scores

主に使用するデータ：

players.csv
appearances.csv
player_valuations.csv

ローカルの data/ にCSVが存在しない場合は、
KaggleHubから自動取得します。

4カ国代表データ

日本・オランダ・スウェーデン・チュニジアの代表データには、
以下のExcelファイルを使用します。

data/world_cup_ai_4countries_japanese.xlsx

このExcelファイルはリポジトリに含まれています。

🚀 ローカル環境での実行
1. リポジトリをclone
git clone https://github.com/Rion-rion/worldcup_predictor_ai.git
cd worldcup_predictor_ai
2. 仮想環境を作成
python -m venv .venv
Windows PowerShell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
macOS / Linux
source .venv/bin/activate
3. ライブラリをインストール
python -m pip install -r requirements.txt
4. 実行
python main.py

正常に完了すると、各処理の最後に完了メッセージが表示されます。

🏆 全処理が正常に完了しました！
🐳 Dockerでの実行

Dockerを利用すると、
ローカルのPython環境に依存せず同一環境で実行できます。

Dockerイメージをビルド
docker build -t worldcup-predictor-ai .
コンテナを実行
docker run --rm worldcup-predictor-ai

Docker Desktop / WSL2環境で動作確認済みです。

Dockerイメージでは、日本語表示に対応するため
Noto Sans CJK フォントを導入しています。

🐳 Docker関連ファイル
Dockerfile

Python 3.13ベースの実行環境を構築します。

主な処理：

Python 3.13環境構築
日本語フォント導入
Pythonライブラリのインストール
プロジェクトファイルのコピー
main.py の実行
.dockerignore

Dockerイメージに不要な以下のファイルを除外します。

.git/
.venv/
__pycache__/
VS Code設定
キャッシュファイル
一時ファイル
ローカルKaggle CSV
旧Colab Notebook
🔄 Colab版からの主な変更

Google Colab版から以下の変更を行いました。

Colab依存のパス処理をローカル環境向けに変更
Pythonコードを機能ごとにファイル分割
requirements.txt による依存関係管理
Python 3.13とのライブラリ互換性問題を修正
japanize-matplotlib 依存を削除
Windows / Linux対応の日本語フォント設定を追加
Git / GitHubによるバージョン管理
.gitignore の追加
.dockerignore の追加
Dockerfileの追加
Docker / WSL2環境での動作確認
⚠️ 注意事項

このプロジェクトの予測結果は、
過去データ・機械学習モデル・確率シミュレーションに基づくものです。

実際の試合結果を保証するものではありません。

また、選手データや市場価値などは取得時点のデータに依存するため、
最新の実際の代表メンバー・コンディションとは異なる場合があります。

🔮 今後の改善候補
対象国を4カ国から48カ国へ拡張
最新FIFAランキングの特徴量追加
選手コンディション・怪我情報の反映
フォーメーション・スタメン情報の反映
モデル精度比較
XGBoost / LightGBMなど別モデルとの比較
GitHub Actionsによる自動テスト
Webアプリ化
Docker Compose対応
API化
📌 Status
✅ Google Colabからローカル環境へ移行
✅ VS Codeで実行確認
✅ Python 3.13対応
✅ GitHub管理
✅ 日本語グラフ表示対応
✅ Dockerfile作成
✅ Dockerイメージのビルド確認
✅ Dockerコンテナ上で正常動作確認
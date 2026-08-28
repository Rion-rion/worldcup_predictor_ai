# ⚽ ワールドカップ予想AI

代表戦データ、Eloレーティング、選手データ、機械学習、モンテカルロシミュレーションを利用して、サッカー代表戦の勝敗・スコア分布・グループリーグ突破確率を予測するPythonプロジェクトです。

このプロジェクトは、Google Colab依存の構成からリファクタリングし、**GitHubからcloneしてローカル環境で実行できる構成**へ移行しています。

---

## 📌 主な機能

- 代表戦データの取得・前処理
- 選手データの取得・前処理
- 選手市場価値・出場時間・得点・アシストを利用した選手スコア算出
- 国別チーム力の算出
- Eloレーティングの計算
- 直近5試合・10試合の特徴量作成
- RandomForestClassifierによる勝敗予測
- ポアソン分布を用いた試合シミュレーション
- 10,000回シミュレーションによるスコア分布算出
- グループリーグ突破確率の算出
- 1分単位の試合ログ生成

---

# 🛠 使用技術

## 言語

- Python

## 開発環境

- Visual Studio Code
- Git / GitHub

## 主なライブラリ

- pandas
- NumPy
- matplotlib
- scikit-learn
- openpyxl
- kagglehub
- joblib
- japanize-matplotlib

## 主なアルゴリズム・手法

- RandomForestClassifier
- Eloレーティング
- ポアソン分布
- モンテカルロシミュレーション

---

# 📁 ディレクトリ構成

```text
worldcup_predictor_ai/
│
├── data/
│   └── world_cup_ai_4countries_japanese.xlsx
│
├── data_prepare.py
├── ai_model.py
├── match_simulation.py
├── japan_group_simulation.py
├── main.py
├── requirements.txt
├── README.md
└── worldcup_predictor_ai_clean_ipynb2.ipynb
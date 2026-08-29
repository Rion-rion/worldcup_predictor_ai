# data/

プロジェクトで使用するデータの配置先です。

## Kaggle選手データ

以下のCSVを使用します。

- `players.csv`
- `appearances.csv`
- `player_valuations.csv`

これらのCSVが `data/` に存在しない場合は、
KaggleHubから以下のデータセットを自動取得します。

```text
davidcariboo/player-scores
4カ国代表データ

日本・オランダ・スウェーデン・チュニジアの代表データには、
以下のExcelファイルを使用します。

data/world_cup_ai_4countries_japanese.xlsx

このExcelファイルはリポジトリに含まれています。

対象国：

Japan
Netherlands
Sweden
Tunisia

各国26名、合計104名の代表選手データを使用します。

補足

Kaggle由来のCSVはローカルキャッシュとして扱うため、
GitHubには含めません。

一方で、
world_cup_ai_4countries_japanese.xlsx は
4カ国代表シミュレーションに必要なデータとして
GitHubリポジトリに含めています。
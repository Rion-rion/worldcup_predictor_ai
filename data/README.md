# data/

ローカルデータの配置先です。

通常の実行ではCSVやExcelの手動配置は必須ではありません。

- `players.csv`
- `appearances.csv`
- `player_valuations.csv`

が無い場合は、KaggleHubから `davidcariboo/player-scores` を自動取得します。

次のExcelは任意です。

```text
data/world_cup_ai_4countries_japanese.xlsx
```

Excelがある場合は、日本・オランダ・スウェーデン・チュニジアの
代表メンバーとしてExcel側を優先します。

Excelが無い場合はKaggleデータの `player_score` 上位26人を
各国の代表候補として自動選出します。

CSV/XLSXは `.gitignore` 対象です。

# ① データ準備
# VS Code / Google Colab 両対応
#
# ・代表戦データ取得
# ・Kaggle選手データ読み込み
# ・選手特徴量作成
# ・4カ国代表Excel読み込み
# ・選手スコア作成

from pathlib import Path
import re
import unicodedata

import kagglehub
import numpy as np
import pandas as pd


# パス設定

try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = BASE_DIR / "data"

SQUAD_FILE = DATA_DIR / "world_cup_ai_4countries_japanese.xlsx"

PLAYERS_FILE = DATA_DIR / "players.csv"
APPEARANCES_FILE = DATA_DIR / "appearances.csv"
VALUATIONS_FILE = DATA_DIR / "player_valuations.csv"

TARGET_TEAMS = [
    "Japan",
    "Netherlands",
    "Sweden",
    "Tunisia",
]


# 国名

COUNTRY_JA = {
    "Japan": "日本",
    "Netherlands": "オランダ",
    "Sweden": "スウェーデン",
    "Tunisia": "チュニジア",
    "Brazil": "ブラジル",
    "Spain": "スペイン",
    "Italy": "イタリア",
    "France": "フランス",
    "Argentina": "アルゼンチン",
    "England": "イングランド",
    "Germany": "ドイツ",
    "Portugal": "ポルトガル",
    "Belgium": "ベルギー",
    "Croatia": "クロアチア",
    "United States": "アメリカ",
    "Korea, South": "韓国",
    "South Korea": "韓国",
    "Mexico": "メキシコ",
    "Morocco": "モロッコ",
    "Uruguay": "ウルグアイ",
    "Switzerland": "スイス",
    "Denmark": "デンマーク",
    "Serbia": "セルビア",
    "Poland": "ポーランド",
    "Norway": "ノルウェー",
    "Turkey": "トルコ",
    "Türkiye": "トルコ",
    "Australia": "オーストラリア",
    "Senegal": "セネガル",
    "Ghana": "ガーナ",
    "Saudi Arabia": "サウジアラビア",
    "Algeria": "アルジェリア",
    "Austria": "オーストリア",
    "Romania": "ルーマニア",
    "Czech Republic": "チェコ",
    "Nigeria": "ナイジェリア",
    "Cote d'Ivoire": "コートジボワール",
    "Côte d'Ivoire": "コートジボワール",
    "Ivory Coast": "コートジボワール",
    "Ireland": "アイルランド",
    "Cameroon": "カメルーン",
    "Bosnia-Herzegovina": "ボスニア・ヘルツェゴビナ",
    "Bosnia and Herzegovina": "ボスニア・ヘルツェゴビナ",
    "Slovakia": "スロバキア",
    "Albania": "アルバニア",
    "Mali": "マリ",
    "Slovenia": "スロベニア",
    "Wales": "ウェールズ",
    "DR Congo": "コンゴ民主共和国",
    "Congo DR": "コンゴ民主共和国",
    "Paraguay": "パラグアイ",
}


def country_ja(team):
    return COUNTRY_JA.get(team, team)


# 共通処理

def remove_duplicate_columns(df):
    return df.loc[:, ~df.columns.duplicated()].copy()


def normalize_name(name):
    if pd.isna(name):
        return ""

    name = str(name)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(ch for ch in name if not unicodedata.combining(ch))
    name = name.lower()
    name = re.sub(r"[^a-z0-9 ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()

    return name


# 代表戦データ

def load_international_results():
    url = (
        "https://raw.githubusercontent.com/"
        "martj42/international_results/"
        "master/results.csv"
    )

    print("代表戦データを読み込んでいます...")

    df = pd.read_csv(url)

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
        ]
    )
    df = df.sort_values("date").reset_index(drop=True)

    def judge(row):
        if row["home_score"] > row["away_score"]:
            return 2
        if row["home_score"] == row["away_score"]:
            return 1
        return 0

    df["result"] = df.apply(judge, axis=1)

    return df


# Kaggleデータ

def load_player_data():
    local_files_exist = all(
        [
            PLAYERS_FILE.exists(),
            APPEARANCES_FILE.exists(),
            VALUATIONS_FILE.exists(),
        ]
    )

    if local_files_exist:
        print("data/ のKaggle CSVを使用します。")

        players = pd.read_csv(PLAYERS_FILE)
        appearances = pd.read_csv(APPEARANCES_FILE)
        valuations = pd.read_csv(VALUATIONS_FILE)

        return players, appearances, valuations

    print("data/ にKaggle CSVがないため、KaggleHubから取得します。")

    kaggle_path = Path(
        kagglehub.dataset_download("davidcariboo/player-scores")
    )

    players = pd.read_csv(kaggle_path / "players.csv")
    appearances = pd.read_csv(kaggle_path / "appearances.csv")
    valuations = pd.read_csv(kaggle_path / "player_valuations.csv")

    return players, appearances, valuations


# Kaggle選手特徴量

def build_player_features(players, appearances, valuations):
    players = remove_duplicate_columns(players.copy())
    appearances = remove_duplicate_columns(appearances.copy())
    valuations = remove_duplicate_columns(valuations.copy())

    minutes_col = None
    for column in ["minutes_played", "minutes", "playing_time"]:
        if column in appearances.columns:
            minutes_col = column
            break

    goals_col = "goals" if "goals" in appearances.columns else None
    assists_col = "assists" if "assists" in appearances.columns else None

    agg_dict = {}

    if minutes_col:
        agg_dict[minutes_col] = "sum"

    if goals_col:
        agg_dict[goals_col] = "sum"

    if assists_col:
        agg_dict[assists_col] = "sum"

    if agg_dict:
        app = appearances.groupby("player_id").agg(agg_dict).reset_index()
    else:
        app = pd.DataFrame({"player_id": players["player_id"]})

    rename_map = {}

    if minutes_col:
        rename_map[minutes_col] = "minutes"

    if goals_col:
        rename_map[goals_col] = "goals"

    if assists_col:
        rename_map[assists_col] = "assists"

    app = app.rename(columns=rename_map)

    for column in ["minutes", "goals", "assists"]:
        if column not in app.columns:
            app[column] = 0

    value_candidates = [
        "market_value_in_eur",
        "market_value",
        "market_value_eur",
        "value",
    ]

    value_col = None

    for column in value_candidates:
        if column in valuations.columns:
            value_col = column
            break

    if value_col is None:
        raise ValueError("市場価値の列が見つかりません。")

    if "date" in valuations.columns:
        valuations["date"] = pd.to_datetime(
            valuations["date"],
            errors="coerce",
        )

        latest_values = (
            valuations.sort_values("date")
            .groupby("player_id")
            .tail(1)[["player_id", value_col]]
        )
    else:
        latest_values = (
            valuations.groupby("player_id")
            .tail(1)[["player_id", value_col]]
        )

    latest_values = latest_values.rename(
        columns={value_col: "market_value_in_eur"}
    )

    df = players.merge(app, on="player_id", how="left")
    df = df.merge(latest_values, on="player_id", how="left")
    df = remove_duplicate_columns(df)

    for column in [
        "minutes",
        "goals",
        "assists",
        "market_value_in_eur",
    ]:
        if column not in df.columns:
            df[column] = 0

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    if "date_of_birth" in df.columns:
        birth = pd.to_datetime(
            df["date_of_birth"],
            errors="coerce",
        )

        df["age"] = 2026 - birth.dt.year
        df["age"] = df["age"].fillna(df["age"].median())
    else:
        df["age"] = 25

    rename_cols = {}

    if "name" in df.columns:
        rename_cols["name"] = "player_name"
    elif "pretty_name" in df.columns:
        rename_cols["pretty_name"] = "player_name"

    if "country_of_citizenship" in df.columns:
        rename_cols["country_of_citizenship"] = "national_team"
    elif "country" in df.columns:
        rename_cols["country"] = "national_team"

    df = df.rename(columns=rename_cols)

    if "player_name" not in df.columns:
        df["player_name"] = "Unknown Player"

    if "national_team" not in df.columns:
        df["national_team"] = "Unknown"

    if "position" not in df.columns:
        df["position"] = "Unknown"

    df["national_team"] = (
        df["national_team"]
        .fillna("Unknown")
        .astype(str)
    )

    df["position"] = (
        df["position"]
        .fillna("Unknown")
        .astype(str)
    )

    df["national_team_ja"] = (
        df["national_team"]
        .map(COUNTRY_JA)
        .fillna(df["national_team"])
    )

    return df


# スコア

def safe_log_score(series, max_point):
    series = pd.to_numeric(
        series,
        errors="coerce",
    ).fillna(0)

    max_value = series.max()

    if max_value <= 0:
        return pd.Series(0, index=series.index)

    return (
        np.log1p(series)
        / np.log1p(max_value)
        * max_point
    )


def add_player_scores(df):
    df = df.copy()

    value_score = safe_log_score(
        df["market_value_in_eur"],
        40,
    )
    minutes_score = safe_log_score(
        df["minutes"],
        25,
    )
    goal_score = safe_log_score(
        df["goals"],
        20,
    )
    assist_score = safe_log_score(
        df["assists"],
        15,
    )

    df["player_score"] = (
        value_score
        + minutes_score
        + goal_score
        + assist_score
    )

    return df


# 4カ国代表Excel

def load_worldcup_squad():
    if not SQUAD_FILE.exists():
        raise FileNotFoundError(
            "\n4カ国代表Excelが見つかりません。\n"
            f"配置先: {SQUAD_FILE}\n"
        )

    print("4カ国代表Excelを読み込んでいます...")

    df = pd.read_excel(
        SQUAD_FILE,
        sheet_name="AI用_All_Players",
        header=2,
        engine="openpyxl",
    )

    df = df.rename(
        columns={
            "Country": "national_team",
            "Player": "player_name",
            "Position": "position",
            "Pos Group": "position_group",
            "Age": "age",
            "Minutes": "minutes",
            "Goals": "goals",
            "Assists": "assists",
            "Season Club(s)": "tm_club",
            "Season": "tm_season",
            "Market Source URL": "tm_source_url",
        }
    )

    required_columns = [
        "national_team",
        "player_name",
        "position",
        "age",
        "minutes",
        "goals",
        "assists",
        "Market Value (€m)",
    ]

    missing = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            "AI用_All_Players に必要な列がありません: "
            + ", ".join(missing)
        )

    df["market_value_in_eur"] = (
        pd.to_numeric(
            df["Market Value (€m)"],
            errors="coerce",
        )
        .fillna(0)
        * 1_000_000
    )

    for column in [
        "minutes",
        "goals",
        "assists",
        "age",
    ]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        ).fillna(0)

    if "position_group" not in df.columns:
        df["position_group"] = np.nan

    df["tm_player_name"] = df["player_name"]
    df["tm_position"] = df["position"]
    df["tm_age"] = df["age"]

    df["is_selected"] = 1

    df["national_team_ja"] = (
        df["national_team"]
        .map(COUNTRY_JA)
        .fillna(df["national_team"])
    )

    df["name_key"] = (
        df["player_name"]
        .apply(normalize_name)
    )

    df = add_player_scores(df)
    df["player_score_original"] = df["player_score"]

    return df


# 全データ準備

def prepare_data():
    print("==========================================")
    print("① データ準備開始")
    print("==========================================")

    intl = load_international_results()

    players, appearances, valuations = load_player_data()

    players_all = build_player_features(
        players,
        appearances,
        valuations,
    )

    players_all = add_player_scores(players_all)
    players_all["is_selected"] = 0

    squad = load_worldcup_squad()

    # 対象4カ国についてはExcel側を正とする
    players_all = players_all[
        ~players_all["national_team"].isin(TARGET_TEAMS)
    ].copy()

    players_scored = pd.concat(
        [
            players_all,
            squad,
        ],
        ignore_index=True,
        sort=False,
    )

    players_scored = remove_duplicate_columns(players_scored)

    print()
    print("==========================================")
    print("データ準備完了")
    print("==========================================")
    print("代表戦:", intl.shape)
    print("代表戦最新日:", intl["date"].max())
    print("全選手:", players_scored.shape)

    selected_count = int(
        players_scored["is_selected"]
        .fillna(0)
        .sum()
    )

    print("4カ国代表:", selected_count, "人")
    print()

    selected_check = (
        players_scored[
            players_scored["national_team"].isin(TARGET_TEAMS)
        ]
        .groupby("national_team")["is_selected"]
        .sum()
    )

    print(selected_check)

    return intl, players_scored


# 単体動作確認

if __name__ == "__main__":
    intl, players_scored = prepare_data()

    selected = players_scored[
        players_scored["is_selected"] == 1
    ][
        [
            "player_name",
            "national_team",
            "position",
            "position_group",
            "age",
            "market_value_in_eur",
            "minutes",
            "goals",
            "assists",
            "player_score",
        ]
    ]

    print()
    print(selected.head(20).to_string(index=False))

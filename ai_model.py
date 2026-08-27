
# ② AIモデル
# Elo + 直近5試合 + 直近10試合
# RandomForestで勝敗予測


from collections import defaultdict

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)


FEATURES_ELO = [
    "home_win_rate_10",
    "away_win_rate_10",
    "home_gf_avg_10",
    "away_gf_avg_10",
    "home_ga_avg_10",
    "away_ga_avg_10",

    "home_win_rate_5",
    "away_win_rate_5",
    "home_gf_avg_5",
    "away_gf_avg_5",
    "home_ga_avg_5",
    "away_ga_avg_5",

    "win_rate_diff_10",
    "gf_diff_10",
    "ga_diff_10",

    "win_rate_diff_5",
    "gf_diff_5",
    "ga_diff_5",

    "home_elo",
    "away_elo",
    "elo_diff",
    "home_expected_by_elo",
    "away_expected_by_elo",

    "neutral",
]


# 直近成績

def get_recent_stats(history, team, date, n=10):
    past = history[
        (
            (history["home_team"] == team)
            | (history["away_team"] == team)
        )
        & (history["date"] < date)
    ].tail(n)

    if len(past) == 0:
        return {
            "win_rate": 0.33,
            "gf_avg": 1.0,
            "ga_avg": 1.0,
        }

    wins = 0
    goals_for = []
    goals_against = []

    for _, row in past.iterrows():
        if row["home_team"] == team:
            team_goals = row["home_score"]
            opponent_goals = row["away_score"]
        else:
            team_goals = row["away_score"]
            opponent_goals = row["home_score"]

        goals_for.append(team_goals)
        goals_against.append(opponent_goals)

        if team_goals > opponent_goals:
            wins += 1

    return {
        "win_rate": wins / len(past),
        "gf_avg": float(np.mean(goals_for)),
        "ga_avg": float(np.mean(goals_against)),
    }


# Elo

def elo_expected_score(rating_a, rating_b):
    return 1 / (
        1
        + 10 ** (
            (rating_b - rating_a)
            / 400
        )
    )


def _safe_neutral(row):
    value = row.get("neutral", 0)

    if pd.isna(value):
        return 0

    if isinstance(value, str):
        return 1 if value.strip().lower() in {
            "1",
            "true",
            "yes",
            "y",
        } else 0

    return int(bool(value))


def add_elo_features(
    intl,
    base_elo=1500,
    k=30,
    home_advantage=60,
):
    df = (
        intl.copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    ratings = defaultdict(lambda: base_elo)

    home_elo_list = []
    away_elo_list = []
    elo_diff_list = []
    home_expected_list = []
    away_expected_list = []

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_elo = ratings[home]
        away_elo = ratings[away]

        neutral = _safe_neutral(row)

        home_adv = (
            0
            if neutral == 1
            else home_advantage
        )

        adjusted_home_elo = home_elo + home_adv

        home_expected = elo_expected_score(
            adjusted_home_elo,
            away_elo,
        )

        away_expected = 1 - home_expected

        home_elo_list.append(home_elo)
        away_elo_list.append(away_elo)

        elo_diff_list.append(
            adjusted_home_elo - away_elo
        )

        home_expected_list.append(
            home_expected
        )

        away_expected_list.append(
            away_expected
        )

        if row["home_score"] > row["away_score"]:
            home_actual = 1.0
            away_actual = 0.0

        elif row["home_score"] == row["away_score"]:
            home_actual = 0.5
            away_actual = 0.5

        else:
            home_actual = 0.0
            away_actual = 1.0

        ratings[home] = (
            home_elo
            + k * (
                home_actual
                - home_expected
            )
        )

        ratings[away] = (
            away_elo
            + k * (
                away_actual
                - away_expected
            )
        )

    df["home_elo"] = home_elo_list
    df["away_elo"] = away_elo_list
    df["elo_diff"] = elo_diff_list
    df["home_expected_by_elo"] = home_expected_list
    df["away_expected_by_elo"] = away_expected_list

    return df


def get_current_elo_ratings(
    intl,
    base_elo=1500,
    k=30,
    home_advantage=60,
):
    ratings = defaultdict(lambda: base_elo)

    df = (
        intl.copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    for _, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]

        home_elo = ratings[home]
        away_elo = ratings[away]

        neutral = _safe_neutral(row)

        home_adv = (
            0
            if neutral == 1
            else home_advantage
        )

        expected_home = elo_expected_score(
            home_elo + home_adv,
            away_elo,
        )

        expected_away = 1 - expected_home

        if row["home_score"] > row["away_score"]:
            actual_home = 1.0
            actual_away = 0.0

        elif row["home_score"] == row["away_score"]:
            actual_home = 0.5
            actual_away = 0.5

        else:
            actual_home = 0.0
            actual_away = 1.0

        ratings[home] = (
            home_elo
            + k * (
                actual_home
                - expected_home
            )
        )

        ratings[away] = (
            away_elo
            + k * (
                actual_away
                - expected_away
            )
        )

    return dict(ratings)


# 学習データ

def build_ml_dataset_with_elo(
    intl,
    since="2023-01-01",
):
    elo_df = add_elo_features(intl)

    since = pd.Timestamp(since)

    base = elo_df[
        elo_df["date"] >= since
    ].copy()

    rows = []

    for _, row in base.iterrows():
        date = row["date"]
        home = row["home_team"]
        away = row["away_team"]

        hist = intl[
            intl["date"] < date
        ]

        home10 = get_recent_stats(
            hist,
            home,
            date,
            n=10,
        )

        away10 = get_recent_stats(
            hist,
            away,
            date,
            n=10,
        )

        home5 = get_recent_stats(
            hist,
            home,
            date,
            n=5,
        )

        away5 = get_recent_stats(
            hist,
            away,
            date,
            n=5,
        )

        rows.append(
            {
                "date": date,
                "home_team": home,
                "away_team": away,

                "home_win_rate_10": home10["win_rate"],
                "away_win_rate_10": away10["win_rate"],
                "home_gf_avg_10": home10["gf_avg"],
                "away_gf_avg_10": away10["gf_avg"],
                "home_ga_avg_10": home10["ga_avg"],
                "away_ga_avg_10": away10["ga_avg"],

                "home_win_rate_5": home5["win_rate"],
                "away_win_rate_5": away5["win_rate"],
                "home_gf_avg_5": home5["gf_avg"],
                "away_gf_avg_5": away5["gf_avg"],
                "home_ga_avg_5": home5["ga_avg"],
                "away_ga_avg_5": away5["ga_avg"],

                "win_rate_diff_10":
                    home10["win_rate"]
                    - away10["win_rate"],

                "gf_diff_10":
                    home10["gf_avg"]
                    - away10["gf_avg"],

                "ga_diff_10":
                    home10["ga_avg"]
                    - away10["ga_avg"],

                "win_rate_diff_5":
                    home5["win_rate"]
                    - away5["win_rate"],

                "gf_diff_5":
                    home5["gf_avg"]
                    - away5["gf_avg"],

                "ga_diff_5":
                    home5["ga_avg"]
                    - away5["ga_avg"],

                "home_elo": row["home_elo"],
                "away_elo": row["away_elo"],
                "elo_diff": row["elo_diff"],

                "home_expected_by_elo":
                    row["home_expected_by_elo"],

                "away_expected_by_elo":
                    row["away_expected_by_elo"],

                "neutral":
                    _safe_neutral(row),

                "result":
                    row["result"],
            }
        )

    return pd.DataFrame(rows)


# モデル学習

def train_model(
    intl,
    since="2023-01-01",
    test_size=0.2,
    show_report=True,
):
    print("==========================================")
    print("② AIモデル学習開始")
    print("==========================================")

    ml_df_elo = build_ml_dataset_with_elo(
        intl,
        since=since,
    )

    ml_df_elo = (
        ml_df_elo
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(ml_df_elo) < 10:
        raise ValueError(
            "学習データが少なすぎます。"
        )

    split_index = int(
        len(ml_df_elo)
        * (1 - test_size)
    )

    split_index = min(
        max(split_index, 1),
        len(ml_df_elo) - 1,
    )

    train_df = ml_df_elo.iloc[
        :split_index
    ].copy()

    test_df = ml_df_elo.iloc[
        split_index:
    ].copy()

    X_train = train_df[FEATURES_ELO]
    y_train = train_df["result"]

    X_test = test_df[FEATURES_ELO]
    y_test = test_df["result"]

    model_elo = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=3,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    model_elo.fit(
        X_train,
        y_train,
    )

    pred_test_elo = model_elo.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        pred_test_elo,
    )

    print()
    print("==========================================")
    print("AIモデル精度")
    print("==========================================")
    print(
        "採用モデル："
        "Elo + 直近5試合 + 直近10試合"
    )
    print(
        "評価方法："
        "古い80%を学習 / 新しい20%をテスト"
    )
    print(
        f"テストデータ正解率："
        f"{accuracy * 100:.2f}%"
    )

    if show_report:
        print("\n分類レポート")

        print(
            classification_report(
                y_test,
                pred_test_elo,
                labels=[0, 1, 2],
                target_names=[
                    "ホーム負け",
                    "引き分け",
                    "ホーム勝ち",
                ],
                zero_division=0,
            )
        )

        print("\n混同行列")

        print(
            confusion_matrix(
                y_test,
                pred_test_elo,
                labels=[0, 1, 2],
            )
        )

    return model_elo, ml_df_elo, accuracy


# 未来の1試合をAIで予測

def build_future_match_features(
    team_a,
    team_b,
    intl,
    neutral=1,
    base_elo=1500,
    home_advantage=60,
):
    intl = intl.copy()

    if not pd.api.types.is_datetime64_any_dtype(
        intl["date"]
    ):
        intl["date"] = pd.to_datetime(
            intl["date"],
            errors="coerce",
        )

    prediction_date = (
        intl["date"].max()
        + pd.Timedelta(days=1)
    )

    home10 = get_recent_stats(
        intl,
        team_a,
        prediction_date,
        n=10,
    )

    away10 = get_recent_stats(
        intl,
        team_b,
        prediction_date,
        n=10,
    )

    home5 = get_recent_stats(
        intl,
        team_a,
        prediction_date,
        n=5,
    )

    away5 = get_recent_stats(
        intl,
        team_b,
        prediction_date,
        n=5,
    )

    ratings = get_current_elo_ratings(
        intl,
        base_elo=base_elo,
        home_advantage=home_advantage,
    )

    home_elo = ratings.get(
        team_a,
        base_elo,
    )

    away_elo = ratings.get(
        team_b,
        base_elo,
    )

    home_adv = (
        0
        if int(neutral) == 1
        else home_advantage
    )

    adjusted_home_elo = (
        home_elo
        + home_adv
    )

    home_expected = elo_expected_score(
        adjusted_home_elo,
        away_elo,
    )

    row = {
        "home_win_rate_10": home10["win_rate"],
        "away_win_rate_10": away10["win_rate"],
        "home_gf_avg_10": home10["gf_avg"],
        "away_gf_avg_10": away10["gf_avg"],
        "home_ga_avg_10": home10["ga_avg"],
        "away_ga_avg_10": away10["ga_avg"],

        "home_win_rate_5": home5["win_rate"],
        "away_win_rate_5": away5["win_rate"],
        "home_gf_avg_5": home5["gf_avg"],
        "away_gf_avg_5": away5["gf_avg"],
        "home_ga_avg_5": home5["ga_avg"],
        "away_ga_avg_5": away5["ga_avg"],

        "win_rate_diff_10":
            home10["win_rate"]
            - away10["win_rate"],

        "gf_diff_10":
            home10["gf_avg"]
            - away10["gf_avg"],

        "ga_diff_10":
            home10["ga_avg"]
            - away10["ga_avg"],

        "win_rate_diff_5":
            home5["win_rate"]
            - away5["win_rate"],

        "gf_diff_5":
            home5["gf_avg"]
            - away5["gf_avg"],

        "ga_diff_5":
            home5["ga_avg"]
            - away5["ga_avg"],

        "home_elo": home_elo,
        "away_elo": away_elo,

        "elo_diff":
            adjusted_home_elo
            - away_elo,

        "home_expected_by_elo":
            home_expected,

        "away_expected_by_elo":
            1 - home_expected,

        "neutral":
            int(neutral),
    }

    return pd.DataFrame(
        [row],
        columns=FEATURES_ELO,
    )


def predict_match_probabilities(
    team_a,
    team_b,
    intl,
    model_elo,
    neutral=1,
):
    features = build_future_match_features(
        team_a,
        team_b,
        intl,
        neutral=neutral,
    )

    raw_probabilities = (
        model_elo.predict_proba(
            features
        )[0]
    )

    class_probabilities = {
        int(label): float(probability)
        for label, probability
        in zip(
            model_elo.classes_,
            raw_probabilities,
        )
    }

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_win_rate":
            round(
                class_probabilities.get(2, 0)
                * 100,
                2,
            ),
        "draw_rate":
            round(
                class_probabilities.get(1, 0)
                * 100,
                2,
            ),
        "team_b_win_rate":
            round(
                class_probabilities.get(0, 0)
                * 100,
                2,
            ),
        "features": features,
    }


# 単体動作確認

if __name__ == "__main__":
    from data_prepare import prepare_data

    intl, _ = prepare_data()

    model_elo, ml_df_elo, accuracy = (
        train_model(intl)
    )

    prediction = predict_match_probabilities(
        "Japan",
        "Netherlands",
        intl,
        model_elo,
        neutral=1,
    )

    print()
    print("日本 vs オランダ AI予測")
    print(prediction)

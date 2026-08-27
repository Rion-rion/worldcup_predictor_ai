# ③ 試合シミュレーション
#
# ・代表26人取得
# ・選手能力からチーム力を計算
# ・期待得点(xG)を計算
# ・Poissonで試合を多数回シミュレーション
# ・②のRandomForest予測も併記
# ・グループリーグシミュレーション
# ・試合ログ生成

from collections import Counter
import random

import japanize_matplotlib  # noqa: F401
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data_prepare import (
    COUNTRY_JA,
    remove_duplicate_columns,
)

from ai_model import (
    predict_match_probabilities,
)


def country_ja(team):
    return COUNTRY_JA.get(
        team,
        team,
    )


# ポジション

def normalize_position(pos):
    if pd.isna(pos):
        return "Unknown"

    pos = str(pos).strip().lower()

    if pos in {
        "gk",
        "goalkeeper",
    }:
        return "GK"

    if (
        "back" in pos
        or "defender" in pos
        or "defence" in pos
        or pos == "df"
    ):
        return "DF"

    if (
        "midfield" in pos
        or pos == "mf"
    ):
        return "MF"

    if (
        "attack" in pos
        or "forward" in pos
        or "winger" in pos
        or "striker" in pos
        or pos == "fw"
    ):
        return "FW"

    return "Unknown"


def _build_position_group(squad):
    squad = squad.copy()

    if "position_group" in squad.columns:
        group = (
            squad["position_group"]
            .astype("object")
            .copy()
        )
    else:
        group = pd.Series(
            np.nan,
            index=squad.index,
            dtype="object",
        )

    if "tm_position" in squad.columns:
        fallback_tm = squad[
            "tm_position"
        ].apply(normalize_position)

        group = group.where(
            group.notna()
            & (
                group.astype(str)
                .str.strip()
                .ne("")
            ),
            fallback_tm,
        )

    if "position" in squad.columns:
        fallback_position = squad[
            "position"
        ].apply(normalize_position)

        group = group.where(
            group.notna()
            & (
                group.astype(str)
                .str.strip()
                .ne("")
            ),
            fallback_position,
        )

    group = (
        group
        .fillna("Unknown")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    group = group.where(
        group.isin(
            [
                "GK",
                "DF",
                "MF",
                "FW",
            ]
        ),
        "Unknown",
    )

    squad["position_group"] = group

    return squad


# 
# 代表メンバー取得

def team_squad(
    players_df,
    team_name,
    top_n=26,
):
    players_df = remove_duplicate_columns(
        players_df
    )

    team = players_df[
        players_df[
            "national_team"
        ]
        .astype(str)
        .str.lower()
        == team_name.lower()
    ].copy()

    if (
        "is_selected" in team.columns
        and team["is_selected"]
        .fillna(0)
        .sum()
        > 0
    ):
        team = team[
            team["is_selected"]
            .fillna(0)
            == 1
        ]

    team = remove_duplicate_columns(
        team
    )

    if "player_score" not in team.columns:
        team["player_score"] = 0

    team["player_score"] = (
        pd.to_numeric(
            team["player_score"],
            errors="coerce",
        )
        .fillna(0)
    )

    team = (
        team
        .sort_values(
            "player_score",
            ascending=False,
        )
        .head(top_n)
        .copy()
    )

    return team


# チーム力

def calculate_team_strength(
    team_name,
    players_df,
):
    squad = team_squad(
        players_df,
        team_name,
        top_n=26,
    )

    squad = remove_duplicate_columns(
        squad
    )

    if len(squad) == 0:
        return {
            "team": team_name,
            "team_ja": country_ja(team_name),
            "squad_size": 0,
            "total_strength": 50.0,
            "attack_strength": 50.0,
            "defense_strength": 50.0,
            "market_value_strength": 0.0,
        }

    squad = _build_position_group(
        squad
    )

    squad["player_score"] = (
        pd.to_numeric(
            squad["player_score"],
            errors="coerce",
        )
        .fillna(0)
    )

    total_strength = float(
        squad["player_score"].mean()
    )

    attack_players = squad[
        squad[
            "position_group"
        ].isin(
            [
                "FW",
                "MF",
            ]
        )
    ]

    if len(attack_players) > 0:
        attack_strength = float(
            attack_players[
                "player_score"
            ].mean()
        )
    else:
        attack_strength = total_strength

    defense_players = squad[
        squad[
            "position_group"
        ].isin(
            [
                "GK",
                "DF",
                "MF",
            ]
        )
    ]

    if len(defense_players) > 0:
        defense_strength = float(
            defense_players[
                "player_score"
            ].mean()
        )
    else:
        defense_strength = total_strength

    if "market_value_in_eur" in squad.columns:
        market_values = (
            pd.to_numeric(
                squad[
                    "market_value_in_eur"
                ],
                errors="coerce",
            )
            .fillna(0)
        )

        market_value_strength = float(
            np.log1p(
                market_values
            ).mean()
        )
    else:
        market_value_strength = 0.0

    return {
        "team": team_name,
        "team_ja": country_ja(team_name),
        "squad_size": len(squad),
        "total_strength": total_strength,
        "attack_strength": attack_strength,
        "defense_strength": defense_strength,
        "market_value_strength":
            market_value_strength,
    }


# 期待得点

def expected_goals(
    team_a,
    team_b,
    players_df,
):
    strength_a = (
        calculate_team_strength(
            team_a,
            players_df,
        )
    )

    strength_b = (
        calculate_team_strength(
            team_b,
            players_df,
        )
    )

    attack_gap_a = (
        strength_a["attack_strength"]
        - strength_b["defense_strength"]
    )

    attack_gap_b = (
        strength_b["attack_strength"]
        - strength_a["defense_strength"]
    )

    base_xg = 1.25

    xg_a = (
        base_xg
        + attack_gap_a * 0.025
    )

    xg_b = (
        base_xg
        + attack_gap_b * 0.025
    )

    market_gap = (
        strength_a[
            "market_value_strength"
        ]
        - strength_b[
            "market_value_strength"
        ]
    )

    xg_a += market_gap * 0.015
    xg_b -= market_gap * 0.015

    xg_a = float(
        np.clip(
            xg_a,
            0.35,
            3.50,
        )
    )

    xg_b = float(
        np.clip(
            xg_b,
            0.35,
            3.50,
        )
    )

    return (
        xg_a,
        xg_b,
        strength_a,
        strength_b,
    )


# 1試合シミュレーション

def simulate_match_many_fast(
    team_a,
    team_b,
    players_df,
    n=10000,
    seed=42,
    intl=None,
    model_elo=None,
    neutral=1,
):
    rng = np.random.default_rng(seed)

    (
        xg_a,
        xg_b,
        strength_a,
        strength_b,
    ) = expected_goals(
        team_a,
        team_b,
        players_df,
    )

    goals_a = rng.poisson(
        xg_a,
        size=n,
    )

    goals_b = rng.poisson(
        xg_b,
        size=n,
    )

    team_a_wins = int(
        np.sum(
            goals_a > goals_b
        )
    )

    draws = int(
        np.sum(
            goals_a == goals_b
        )
    )

    team_b_wins = int(
        np.sum(
            goals_a < goals_b
        )
    )

    score_counter = Counter(
        zip(
            goals_a.tolist(),
            goals_b.tolist(),
        )
    )

    most_common_score = (
        score_counter
        .most_common(1)[0][0]
    )

    ai_prediction = None

    if (
        intl is not None
        and model_elo is not None
    ):
        ai_prediction = (
            predict_match_probabilities(
                team_a,
                team_b,
                intl,
                model_elo,
                neutral=neutral,
            )
        )

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_ja": country_ja(team_a),
        "team_b_ja": country_ja(team_b),

        "xg_a": round(xg_a, 2),
        "xg_b": round(xg_b, 2),

        "team_a_win_rate":
            round(
                team_a_wins
                / n
                * 100,
                2,
            ),

        "draw_rate":
            round(
                draws
                / n
                * 100,
                2,
            ),

        "team_b_win_rate":
            round(
                team_b_wins
                / n
                * 100,
                2,
            ),

        "most_common_score":
            (
                f"{most_common_score[0]}"
                f"-"
                f"{most_common_score[1]}"
            ),

        "score_counter":
            score_counter,

        "strength_a":
            strength_a,

        "strength_b":
            strength_b,

        "ai_prediction":
            ai_prediction,

        "n":
            n,
    }


# 試合結果表示

def show_match_result_ja(
    result,
    top_n=12,
    show_graph=True,
):
    print("==========================================")
    print(
        f"{result['team_a_ja']} "
        f"vs "
        f"{result['team_b_ja']} "
        "試合シミュレーション"
    )
    print("==========================================")

    print(
        f"{result['team_a_ja']} "
        f"勝利確率: "
        f"{result['team_a_win_rate']}%"
    )

    print(
        f"引き分け確率: "
        f"{result['draw_rate']}%"
    )

    print(
        f"{result['team_b_ja']} "
        f"勝利確率: "
        f"{result['team_b_win_rate']}%"
    )

    print(
        f"{result['team_a_ja']} "
        f"期待得点: "
        f"{result['xg_a']}"
    )

    print(
        f"{result['team_b_ja']} "
        f"期待得点: "
        f"{result['xg_b']}"
    )

    print(
        f"最頻スコア: "
        f"{result['most_common_score']}"
    )

    ai_prediction = result.get(
        "ai_prediction"
    )

    if ai_prediction is not None:
        print()
        print("② RandomForest予測")

        print(
            f"{result['team_a_ja']} "
            f"勝利: "
            f"{ai_prediction['team_a_win_rate']}%"
        )

        print(
            "引き分け: "
            f"{ai_prediction['draw_rate']}%"
        )

        print(
            f"{result['team_b_ja']} "
            f"勝利: "
            f"{ai_prediction['team_b_win_rate']}%"
        )

    print()

    score_df = pd.DataFrame(
        [
            {
                "スコア": f"{a}-{b}",
                "回数": count,
                "確率": round(
                    count
                    / result["n"]
                    * 100,
                    2,
                ),
            }
            for (
                a,
                b,
            ), count
            in result[
                "score_counter"
            ].most_common(
                top_n
            )
        ]
    )

    print(
        score_df.to_string(
            index=False
        )
    )

    if show_graph:
        plt.figure(
            figsize=(10, 5)
        )

        plt.bar(
            score_df["スコア"],
            score_df["確率"],
        )

        plt.title(
            f"{result['team_a_ja']} "
            f"vs "
            f"{result['team_b_ja']} "
            f"スコア分布 TOP{top_n}"
        )

        plt.xlabel("スコア")
        plt.ylabel("確率（%）")

        plt.xticks(
            rotation=45
        )

        plt.tight_layout()
        plt.show()

    return score_df


# グループリーグ

def simulate_group_many_fast(
    group_teams,
    players_df,
    n=10000,
    seed=42,
):
    rng = np.random.default_rng(
        seed
    )

    group_teams = list(
        group_teams
    )

    standings_count = {
        team: {
            "first": 0,
            "second": 0,
            "qualified": 0,
            "points_total": 0,
            "gd_total": 0,
            "gf_total": 0,
        }
        for team in group_teams
    }

    match_pairs = []

    for i in range(
        len(group_teams)
    ):
        for j in range(
            i + 1,
            len(group_teams),
        ):
            match_pairs.append(
                (
                    group_teams[i],
                    group_teams[j],
                )
            )

    xg_map = {}

    for team_a, team_b in match_pairs:
        xg_a, xg_b, _, _ = (
            expected_goals(
                team_a,
                team_b,
                players_df,
            )
        )

        xg_map[
            (
                team_a,
                team_b,
            )
        ] = (
            xg_a,
            xg_b,
        )

    for _ in range(n):
        table = {
            team: {
                "points": 0,
                "gf": 0,
                "ga": 0,
                "gd": 0,
                "wins": 0,
                "draws": 0,
                "losses": 0,
            }
            for team in group_teams
        }

        for (
            team_a,
            team_b,
        ) in match_pairs:
            xg_a, xg_b = xg_map[
                (
                    team_a,
                    team_b,
                )
            ]

            goals_a = int(
                rng.poisson(xg_a)
            )

            goals_b = int(
                rng.poisson(xg_b)
            )

            table[team_a]["gf"] += (
                goals_a
            )

            table[team_a]["ga"] += (
                goals_b
            )

            table[team_b]["gf"] += (
                goals_b
            )

            table[team_b]["ga"] += (
                goals_a
            )

            if goals_a > goals_b:
                table[team_a]["points"] += 3
                table[team_a]["wins"] += 1
                table[team_b]["losses"] += 1

            elif goals_a < goals_b:
                table[team_b]["points"] += 3
                table[team_b]["wins"] += 1
                table[team_a]["losses"] += 1

            else:
                table[team_a]["points"] += 1
                table[team_b]["points"] += 1
                table[team_a]["draws"] += 1
                table[team_b]["draws"] += 1

        for team in group_teams:
            table[team]["gd"] = (
                table[team]["gf"]
                - table[team]["ga"]
            )

        # 同率時の最後の判定は乱数。
        # 大会正式ルールを実装する場合は今後ここを置き換える。
        tie_break = {
            team: float(
                rng.random()
            )
            for team in group_teams
        }

        ranking = sorted(
            group_teams,
            key=lambda t: (
                table[t]["points"],
                table[t]["gd"],
                table[t]["gf"],
                tie_break[t],
            ),
            reverse=True,
        )

        for idx, team in enumerate(
            ranking
        ):
            if idx == 0:
                standings_count[
                    team
                ]["first"] += 1

            if idx == 1:
                standings_count[
                    team
                ]["second"] += 1

            if idx <= 1:
                standings_count[
                    team
                ]["qualified"] += 1

            standings_count[
                team
            ]["points_total"] += (
                table[team]["points"]
            )

            standings_count[
                team
            ]["gd_total"] += (
                table[team]["gd"]
            )

            standings_count[
                team
            ]["gf_total"] += (
                table[team]["gf"]
            )

    rows = []

    for team in group_teams:
        rows.append(
            {
                "国":
                    country_ja(team),

                "team":
                    team,

                "1位確率":
                    round(
                        standings_count[
                            team
                        ]["first"]
                        / n
                        * 100,
                        2,
                    ),

                "2位確率":
                    round(
                        standings_count[
                            team
                        ]["second"]
                        / n
                        * 100,
                        2,
                    ),

                "突破率":
                    round(
                        standings_count[
                            team
                        ]["qualified"]
                        / n
                        * 100,
                        2,
                    ),

                "平均勝ち点":
                    round(
                        standings_count[
                            team
                        ]["points_total"]
                        / n,
                        2,
                    ),

                "平均得失点差":
                    round(
                        standings_count[
                            team
                        ]["gd_total"]
                        / n,
                        2,
                    ),

                "平均得点":
                    round(
                        standings_count[
                            team
                        ]["gf_total"]
                        / n,
                        2,
                    ),
            }
        )

    result_df = pd.DataFrame(
        rows
    )

    result_df = (
        result_df
        .sort_values(
            "突破率",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result_df


def show_group_result_ja(
    group_result_df,
    show_graph=True,
):
    print("==========================================")
    print("グループリーグ突破率シミュレーション")
    print("==========================================")

    print(
        group_result_df.to_string(
            index=False
        )
    )

    if show_graph:
        plt.figure(
            figsize=(9, 5)
        )

        plt.bar(
            group_result_df["国"],
            group_result_df["突破率"],
        )

        plt.title(
            "グループリーグ突破率"
        )

        plt.xlabel("国")
        plt.ylabel("突破率（%）")
        plt.ylim(0, 100)

        plt.tight_layout()
        plt.show()


# 試合ログ

def get_team_players_for_sim(
    players_df,
    team_name,
):
    squad = team_squad(
        players_df,
        team_name,
        top_n=26,
    )

    squad = remove_duplicate_columns(
        squad
    )

    if len(squad) == 0:
        return pd.DataFrame()

    squad = _build_position_group(
        squad
    )

    squad["player_score"] = (
        pd.to_numeric(
            squad["player_score"],
            errors="coerce",
        )
        .fillna(1)
    )

    return squad


def weighted_choice_players(
    players_df,
    position_groups=None,
):
    if len(players_df) == 0:
        return "Unknown Player"

    df = players_df.copy()

    if position_groups is not None:
        filtered = df[
            df[
                "position_group"
            ].isin(
                position_groups
            )
        ]

        if len(filtered) > 0:
            df = filtered

    weights = (
        pd.to_numeric(
            df["player_score"],
            errors="coerce",
        )
        .fillna(1)
        .clip(lower=1)
    )

    chosen = df.sample(
        1,
        weights=weights,
        replace=True,
    )

    row = chosen.iloc[0]

    if (
        "tm_player_name"
        in chosen.columns
        and pd.notna(
            row.get(
                "tm_player_name"
            )
        )
    ):
        return row[
            "tm_player_name"
        ]

    return row[
        "player_name"
    ]


def simulate_live_match(
    team_a,
    team_b,
    players_df,
    seed=42,
):
    rng = np.random.default_rng(
        seed
    )

    random.seed(seed)

    (
        xg_a,
        xg_b,
        _,
        _,
    ) = expected_goals(
        team_a,
        team_b,
        players_df,
    )

    goals_a = int(
        rng.poisson(xg_a)
    )

    goals_b = int(
        rng.poisson(xg_b)
    )

    players_a = (
        get_team_players_for_sim(
            players_df,
            team_a,
        )
    )

    players_b = (
        get_team_players_for_sim(
            players_df,
            team_b,
        )
    )

    events = [
        {
            "minute": 0,
            "team": "",
            "event": "試合開始",
            "score": "0-0",
        }
    ]

    score_a = 0
    score_b = 0

    total_goals = (
        goals_a
        + goals_b
    )

    goal_minutes = sorted(
        random.randint(1, 89)
        for _ in range(
            total_goals
        )
    )

    goal_events = (
        [team_a] * goals_a
        + [team_b] * goals_b
    )

    random.shuffle(
        goal_events
    )

    for (
        minute,
        scoring_team,
    ) in zip(
        goal_minutes,
        goal_events,
    ):
        if scoring_team == team_a:
            scorer = (
                weighted_choice_players(
                    players_a,
                    ["FW", "MF"],
                )
            )

            score_a += 1

            events.append(
                {
                    "minute": minute,
                    "team":
                        country_ja(
                            team_a
                        ),
                    "event":
                        f"{scorer} がゴール！",
                    "score":
                        f"{score_a}-{score_b}",
                }
            )

        else:
            scorer = (
                weighted_choice_players(
                    players_b,
                    ["FW", "MF"],
                )
            )

            score_b += 1

            events.append(
                {
                    "minute": minute,
                    "team":
                        country_ja(
                            team_b
                        ),
                    "event":
                        f"{scorer} がゴール！",
                    "score":
                        f"{score_a}-{score_b}",
                }
            )

    extra_minutes = sorted(
        random.sample(
            range(1, 90),
            10,
        )
    )

    for minute in extra_minutes:
        event_team = random.choice(
            [
                team_a,
                team_b,
            ]
        )

        if event_team == team_a:
            player = (
                weighted_choice_players(
                    players_a,
                    ["FW", "MF"],
                )
            )

            team_label = country_ja(
                team_a
            )

        else:
            player = (
                weighted_choice_players(
                    players_b,
                    ["FW", "MF"],
                )
            )

            team_label = country_ja(
                team_b
            )

        event_text = random.choice(
            [
                f"{player} がシュート",
                f"{player} がチャンスを作る",
                f"{player} がドリブルで仕掛ける",
                f"{player} が決定機に絡む",
                f"{player} がクロスを上げる",
            ]
        )

        current_score_a = 0
        current_score_b = 0

        for event in events:
            if (
                event["minute"]
                <= minute
                and "ゴール"
                in event["event"]
            ):
                if event["team"] == country_ja(
                    team_a
                ):
                    current_score_a += 1
                elif event["team"] == country_ja(
                    team_b
                ):
                    current_score_b += 1

        events.append(
            {
                "minute": minute,
                "team": team_label,
                "event": event_text,
                "score":
                    (
                        f"{current_score_a}"
                        f"-"
                        f"{current_score_b}"
                    ),
            }
        )

    events.append(
        {
            "minute": 90,
            "team": "",
            "event": "試合終了",
            "score":
                f"{score_a}-{score_b}",
        }
    )

    events = sorted(
        events,
        key=lambda x: x["minute"],
    )

    return {
        "team_a": team_a,
        "team_b": team_b,
        "team_a_ja":
            country_ja(team_a),
        "team_b_ja":
            country_ja(team_b),
        "score":
            f"{score_a}-{score_b}",
        "events":
            events,
    }


def show_live_match_result(
    live_result,
):
    print("==========================================")
    print(
        f"{live_result['team_a_ja']} "
        f"vs "
        f"{live_result['team_b_ja']} "
        "試合ログ"
    )
    print("==========================================")

    print(
        "最終スコア: "
        f"{live_result['team_a_ja']} "
        f"{live_result['score']} "
        f"{live_result['team_b_ja']}"
    )

    print()

    log_df = pd.DataFrame(
        live_result["events"]
    )

    print(
        log_df.to_string(
            index=False
        )
    )

    return log_df


# 単体動作確認

if __name__ == "__main__":
    from data_prepare import prepare_data
    from ai_model import train_model

    intl, players_scored = (
        prepare_data()
    )

    model_elo, _, _ = (
        train_model(
            intl,
            show_report=False,
        )
    )

    print()
    print("==========================================")
    print("③ 試合シミュレーション準備完了")
    print("==========================================")

    for team in [
        "Japan",
        "Netherlands",
        "Sweden",
        "Tunisia",
    ]:
        squad = team_squad(
            players_scored,
            team,
            top_n=26,
        )

        print(
            f"{country_ja(team)}: "
            f"{len(squad)}人"
        )

    result = simulate_match_many_fast(
        "Japan",
        "Netherlands",
        players_scored,
        n=10000,
        seed=42,
        intl=intl,
        model_elo=model_elo,
        neutral=1,
    )

    show_match_result_ja(
        result
    )

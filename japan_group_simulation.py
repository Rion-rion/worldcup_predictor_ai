import matplotlib.pyplot as plt
import pandas as pd

from match_simulation import (
    simulate_group_many_fast,
    simulate_match_many_fast,
    show_group_result_ja,
)


GROUP_TEAMS = [
    "Japan",
    "Netherlands",
    "Sweden",
    "Tunisia",
]


GROUP_MATCHES = [
    ("Japan", "Netherlands"),
    ("Japan", "Sweden"),
    ("Japan", "Tunisia"),
    ("Netherlands", "Sweden"),
    ("Netherlands", "Tunisia"),
    ("Sweden", "Tunisia"),
]


def run_japan_group_simulation(
    players_scored,
    intl=None,
    model_elo=None,
    n=10000,
    base_seed=42,
    show_graphs=True,
    run_group_table=True,
):

    print()
    print("=" * 60)
    print("④ グループリーグ 全6試合予測")
    print("=" * 60)

    # ① グループ全6試合を予測
    all_match_results = []

    for index, (team_a, team_b) in enumerate(
        GROUP_MATCHES
    ):
        result = simulate_match_many_fast(
            team_a,
            team_b,
            players_scored,
            n=n,
            seed=base_seed + index,
            intl=intl,
            model_elo=model_elo,
            neutral=1,
        )

        all_match_results.append(result)

    # ② 全6試合を表にまとめる
    rows = []

    for result in all_match_results:
        ai_prediction = result.get(
            "ai_prediction"
        )

        row = {
            "対戦カード": (
                f"{result['team_a_ja']} "
                f"vs "
                f"{result['team_b_ja']}"
            ),
            "チームA勝利_Poisson":
                result["team_a_win_rate"],
            "引き分け_Poisson":
                result["draw_rate"],
            "チームB勝利_Poisson":
                result["team_b_win_rate"],
            "期待得点A":
                result["xg_a"],
            "期待得点B":
                result["xg_b"],
            "最頻スコア":
                result["most_common_score"],
        }

        if ai_prediction is not None:
            row.update(
                {
                    "チームA勝利_AI":
                        ai_prediction[
                            "team_a_win_rate"
                        ],
                    "引き分け_AI":
                        ai_prediction[
                            "draw_rate"
                        ],
                    "チームB勝利_AI":
                        ai_prediction[
                            "team_b_win_rate"
                        ],
                }
            )

        rows.append(row)

    all_matches_df = pd.DataFrame(rows)

    print()
    print("=" * 60)
    print("グループリーグ 全6試合まとめ")
    print("=" * 60)

    print(
        all_matches_df.to_string(
            index=False
        )
    )

    # ③ 日本の3試合を抜き出す
    japan_rows = []

    for result in all_match_results:

        if result["team_a"] != "Japan":
            continue

        ai_prediction = result.get(
            "ai_prediction"
        )

        row = {
            "相手":
                result["team_b_ja"],

            "日本勝利_Poisson":
                result["team_a_win_rate"],

            "引き分け_Poisson":
                result["draw_rate"],

            "日本敗北_Poisson":
                result["team_b_win_rate"],
        }

        if ai_prediction is not None:
            row.update(
                {
                    "日本勝利_AI":
                        ai_prediction[
                            "team_a_win_rate"
                        ],

                    "引き分け_AI":
                        ai_prediction[
                            "draw_rate"
                        ],

                    "日本敗北_AI":
                        ai_prediction[
                            "team_b_win_rate"
                        ],
                }
            )

        japan_rows.append(row)

    japan_match_df = pd.DataFrame(
        japan_rows
    )

    print()
    print("=" * 60)
    print("🇯🇵 日本 3試合の勝率")
    print("=" * 60)

    print(
        japan_match_df.to_string(
            index=False
        )
    )

    # ④ 日本3試合の勝率グラフ
    if show_graphs:

        labels = (
            japan_match_df["相手"]
            .astype(str)
            .tolist()
        )

        japan_rates = (
            japan_match_df[
                "日本勝利_Poisson"
            ].tolist()
        )

        draw_rates = (
            japan_match_df[
                "引き分け_Poisson"
            ].tolist()
        )

        lose_rates = (
            japan_match_df[
                "日本敗北_Poisson"
            ].tolist()
        )

        x = range(len(labels))
        width = 0.25

        plt.figure(
            figsize=(10, 5)
        )

        plt.bar(
            [
                i - width
                for i in x
            ],
            japan_rates,
            width=width,
            label="日本勝利",
        )

        plt.bar(
            x,
            draw_rates,
            width=width,
            label="引き分け",
        )

        plt.bar(
            [
                i + width
                for i in x
            ],
            lose_rates,
            width=width,
            label="日本敗北",
        )

        plt.xticks(
            list(x),
            labels,
        )

        plt.ylabel(
            "確率（%）"
        )

        plt.title(
            "日本 グループリーグ3試合 勝敗予測"
        )

        plt.ylim(
            0,
            100,
        )

        plt.legend()
        plt.tight_layout()

    # ⑤ グループリーグ突破率
    group_result_df = None

    if run_group_table:

        print()
        print("=" * 60)
        print("🏆 グループリーグ突破率")
        print("=" * 60)

        group_result_df = (
            simulate_group_many_fast(
                GROUP_TEAMS,
                players_scored,
                n=n,
                seed=base_seed,
            )
        )

        show_group_result_ja(
            group_result_df,
            show_graph=show_graphs,
        )

        japan_group_result = (
            group_result_df[
                group_result_df[
                    "team"
                ]
                == "Japan"
            ]
        )

        if not japan_group_result.empty:

            japan_row = (
                japan_group_result
                .iloc[0]
            )

            print()
            print("=" * 60)
            print("🇯🇵 日本代表 最終予測")
            print("=" * 60)

            print(
                f"1位通過率: "
                f"{japan_row['1位確率']}%"
            )

            print(
                f"2位通過率: "
                f"{japan_row['2位確率']}%"
            )

            print(
                f"グループ突破率: "
                f"{japan_row['突破率']}%"
            )

            print(
                f"平均勝ち点: "
                f"{japan_row['平均勝ち点']}"
            )

    return (
        japan_match_df,
        all_match_results,
        group_result_df,
    )


# 単体動作確認
if __name__ == "__main__":

    from data_prepare import (
        prepare_data,
    )

    from ai_model import (
        train_model,
    )

    intl, players_scored = (
        prepare_data()
    )

    (
        model_elo,
        _,
        _,
    ) = train_model(
        intl
    )

    run_japan_group_simulation(
        players_scored=players_scored,
        intl=intl,
        model_elo=model_elo,
        n=10000,
        base_seed=42,
        show_graphs=True,
        run_group_table=True,
    )

    plt.show()
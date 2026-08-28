# ④ 日本のグループ3試合
#
# ・日本 vs オランダ
# ・日本 vs スウェーデン
# ・日本 vs チュニジア
# ・③のシミュレーションを1試合につき1回だけ実行
# ・Poisson予測 + ②RandomForest予測を表示

import matplotlib.pyplot as plt
import pandas as pd

from data_prepare import country_ja

from match_simulation import (
    simulate_group_many_fast,
    simulate_match_many_fast,
    show_group_result_ja,
    show_match_result_ja,
)


JAPAN_GROUP_MATCHES = [
    ("Japan", "Netherlands"),
    ("Japan", "Sweden"),
    ("Japan", "Tunisia"),
]

GROUP_TEAMS = [
    "Japan",
    "Netherlands",
    "Sweden",
    "Tunisia",
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
    print("==========================================")
    print("④ 日本グループ3試合シミュレーション")
    print("==========================================")

    results = []

    for index, (
        team_a,
        team_b,
    ) in enumerate(
        JAPAN_GROUP_MATCHES
    ):
        seed = (
            base_seed
            + index
        )

        result = (
            simulate_match_many_fast(
                team_a,
                team_b,
                players_scored,
                n=n,
                seed=seed,
                intl=intl,
                model_elo=model_elo,
                neutral=1,
            )
        )

        results.append(
            result
        )

    rows = []

    for result in results:
        ai_prediction = (
            result.get(
                "ai_prediction"
            )
        )

        row = {
            "対戦カード":
                (
                    f"{result['team_a_ja']} "
                    f"vs "
                    f"{result['team_b_ja']}"
                ),

            "日本勝利_Poisson":
                result[
                    "team_a_win_rate"
                ],

            "引き分け_Poisson":
                result[
                    "draw_rate"
                ],

            "相手勝利_Poisson":
                result[
                    "team_b_win_rate"
                ],

            "相手":
                result[
                    "team_b_ja"
                ],

            "日本期待得点":
                result[
                    "xg_a"
                ],

            "相手期待得点":
                result[
                    "xg_b"
                ],

            "最頻スコア":
                result[
                    "most_common_score"
                ],
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

                    "相手勝利_AI":
                        ai_prediction[
                            "team_b_win_rate"
                        ],
                }
            )

        rows.append(
            row
        )

    japan_match_df = (
        pd.DataFrame(
            rows
        )
    )

    print()
    print("==========================================")
    print("日本のグループ3試合まとめ")
    print("==========================================")

    print(
        japan_match_df.to_string(
            index=False
        )
    )

    # すでに計算済みのresultsを使うため、
    # 3試合をもう一度シミュレーションしない。
    for result in results:
        print()

        show_match_result_ja(
            result,
            show_graph=show_graphs,
        )

    if show_graphs:
        labels = (
            japan_match_df[
                "相手"
            ]
            .astype(str)
            .tolist()
        )

        japan_rates = (
            japan_match_df[
                "日本勝利_Poisson"
            ]
            .tolist()
        )

        draw_rates = (
            japan_match_df[
                "引き分け_Poisson"
            ]
            .tolist()
        )

        opponent_rates = (
            japan_match_df[
                "相手勝利_Poisson"
            ]
            .tolist()
        )

        x = range(
            len(labels)
        )

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
            opponent_rates,
            width=width,
            label="相手勝利",
        )

        plt.xticks(
            list(x),
            labels,
        )

        plt.ylabel(
            "確率（%）"
        )

        plt.title(
            "日本 グループ3試合 Poisson予測"
        )

        plt.ylim(
            0,
            100,
        )

        plt.legend()
        plt.tight_layout()
        plt.show()

    group_result_df = None

    if run_group_table:
        print()
        print("==========================================")
        print("4カ国グループリーグ全体")
        print("==========================================")

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

    return (
        japan_match_df,
        results,
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

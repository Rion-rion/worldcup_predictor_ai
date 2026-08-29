# main.py
# ワールドカップ予想AI エントリーポイント

import argparse

from ai_model import train_model
from data_prepare import prepare_data
from japan_group_simulation import run_japan_group_simulation
from match_simulation import (
    show_live_match_result,
    simulate_live_match,
)


def run_pipeline(
    show_graphs=False,
    simulations=10000,
):
    """
    ワールドカップ予想AIの全処理を実行する。

    処理順:
    1. データ準備
    2. AIモデル学習
    3. 試合ログ生成
    4. グループリーグシミュレーション
    """

    if simulations <= 0:
        raise ValueError(
            "simulations は1以上を指定してください。"
        )

    print()
    print("=" * 60)
    print("⚽ World Cup Predictor AI")
    print("=" * 60)
    print()

    # ==========================================
    # ① データ準備
    # ==========================================

    intl, players_scored = prepare_data()

    # ==========================================
    # ② AIモデル学習
    # ==========================================

    (
        model_elo,
        _,
        accuracy,
    ) = train_model(
        intl,
        show_report=True,
    )

    # ==========================================
    # ③ 試合ログ生成
    # ==========================================

    print()
    print("=" * 60)
    print("③ 日本 vs オランダ 試合ログ")
    print("=" * 60)

    live_result = simulate_live_match(
        "Japan",
        "Netherlands",
        players_scored,
        seed=42,
    )

    live_log_df = show_live_match_result(
        live_result
    )

    # ==========================================
    # ④ グループリーグ
    # ==========================================

    print()
    print("=" * 60)
    print("④ グループリーグシミュレーション")
    print("=" * 60)

    (
        japan_match_df,
        match_results,
        group_result_df,
    ) = run_japan_group_simulation(
        players_scored=players_scored,
        intl=intl,
        model_elo=model_elo,
        n=simulations,
        base_seed=42,
        show_graphs=show_graphs,
        run_group_table=True,
    )

    # ==========================================
    # 完了
    # ==========================================

    print()
    print("=" * 60)
    print("🏆 全処理が正常に完了しました！")
    print("=" * 60)

    print(
        f"AIモデル精度: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"シミュレーション回数: "
        f"{simulations:,}回"
    )

    # 今後テストやAPI化するときにも利用できるよう
    # 結果をまとめて返す
    return {
        "model_accuracy": accuracy,
        "live_result": live_result,
        "live_log": live_log_df,
        "japan_matches": japan_match_df,
        "match_results": match_results,
        "group_result": group_result_df,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "World Cup Predictor AI"
        )
    )

    parser.add_argument(
        "--graphs",
        action="store_true",
        help=(
            "Matplotlibのグラフを表示する"
        ),
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=10000,
        help=(
            "モンテカルロシミュレーション回数 "
            "(default: 10000)"
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_pipeline(
        show_graphs=args.graphs,
        simulations=args.simulations,
    )


if __name__ == "__main__":
    main()
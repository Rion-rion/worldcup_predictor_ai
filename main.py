import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "data_prepare.py",
    "ai_model.py",
    "match_simulation.py",
    "japan_group_simulation.py",
]


def run_script(script_name):
    script_path = BASE_DIR / script_name

    if not script_path.exists():
        print(f"❌ ファイルが見つかりません: {script_name}")
        return False

    print("\n" + "=" * 60)
    print(f"▶ {script_name} を実行します")
    print("=" * 60)

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        print(f"\n❌ {script_name} でエラーが発生しました。")
        return False

    print(f"\n✅ {script_name} 完了")
    return True


def main():
    print("\n⚽ ワールドカップ予想AI")
    print("プログラムを開始します。\n")

    for script in SCRIPTS:
        success = run_script(script)

        if not success:
            print("\n⚠️ エラーが発生したため処理を停止します。")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("🏆 全処理が正常に完了しました！")
    print("=" * 60)


if __name__ == "__main__":
    main()
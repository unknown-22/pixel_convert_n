"""
ドット絵変換ツールのメインモジュール
UIとロジック部分を分離し、このファイルからアプリケーションを起動する
"""

from ui import create_ui


def main() -> None:
    """
    メイン関数: アプリケーションのエントリーポイント
    Gradioインターフェースを作成して起動する
    """
    interface = create_ui()
    interface.queue(default_concurrency_limit=1)
    interface.launch(share=False)


if __name__ == "__main__":
    main()

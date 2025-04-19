"""
ドット絵変換ツールのメインモジュール
UIとロジック部分を分離し、このファイルからアプリケーションを起動する
"""

import asyncio

from ui import create_ui


async def main():
    """
    メイン関数: アプリケーションのエントリーポイント
    Gradioインターフェースを作成して起動する
    """
    interface = create_ui()
    interface.queue()
    interface.launch(share=False)


if __name__ == "__main__":
    asyncio.run(main())

"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 会話形式で質問する

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors
from google.genai.chats import Chat
from google.genai.types import GenerateContentResponse

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"


def print_usage(response: GenerateContentResponse) -> None:
    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[tokens] input={usage.prompt_token_count}, "
            f"output={usage.candidates_token_count}"
        )


def ask(question: str, chat: Chat | None = None) -> str:
    if chat is None:
        client = genai.Client(api_key=get_api_key())
        started_at = time.perf_counter()
        response = client.models.generate_content(model=MODEL, contents=question)
    else:
        started_at = time.perf_counter()
        response = chat.send_message(question)
    elapsed = time.perf_counter() - started_at

    print_usage(response)
    print(f"[elapsed] {elapsed:.2f} s")
    return response.text


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.question is not None:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
            return

        client = genai.Client(api_key=get_api_key())
        chat = client.chats.create(model=MODEL)
        while True:
            question = input("[質問] ")
            if question.strip().lower() == "exit":
                break
            if not question.strip():
                continue

            print()
            print(ask(question, chat))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()

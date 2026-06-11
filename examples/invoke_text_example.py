from __future__ import annotations

from aws_bedrock_invoke import BedrockInvoker


def main() -> None:
    invoker = BedrockInvoker.from_env()

    result = invoker.invoke_text(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        profile_arn_fmt="arn:aws:bedrock:{region}:354173731728:inference-profile/{model}",
        system_prompt="Answer briefly.",
        user_content=[
            {"type": "text", "text": "What is a lien waiver?"},
        ],
    )

    print(result.response_text)


if __name__ == "__main__":
    main()

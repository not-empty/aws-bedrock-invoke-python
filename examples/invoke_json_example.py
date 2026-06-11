from __future__ import annotations

from aws_bedrock_invoke import BedrockInvoker, json_text_block


def main() -> None:
    invoker = BedrockInvoker.from_env()

    result = invoker.invoke_json(
        model="us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        profile_arn_fmt="arn:aws:bedrock:{region}:354173731728:inference-profile/{model}",
        system_prompt="Reply with strict JSON.",
        user_content=[
            json_text_block({"question": "hello"}),
        ],
    )

    print("parsed_json:")
    print(result.parsed_json)
    print()
    print("response_text:")
    print(result.response_text)


if __name__ == "__main__":
    main()

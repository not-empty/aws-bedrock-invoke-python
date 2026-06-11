from __future__ import annotations


def resolve_model_id(model: str, *, region: str, profile_arn_fmt: str | None = None) -> str:
    if not model:
        raise ValueError("model must not be empty")
    if model.startswith("arn:"):
        return model
    if model.startswith(("us.", "eu.", "apac.", "global.", "au.")):
        if not profile_arn_fmt:
            raise ValueError(
                "profile_arn_fmt is required when model uses an inference-profile shorthand."
            )
        return profile_arn_fmt.format(region=region, model=model)
    return model

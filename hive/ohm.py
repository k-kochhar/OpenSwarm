from llms import oai, gog, cla

MODEL_MAP = {
    "gpt": oai,
    "gemini": gog,
    "claude": cla,
}


def chat(model: str, message: str, image_b64: str = None) -> str:
    for key, module in MODEL_MAP.items():
        if key in model.lower():
            kwargs = {"message": message, "model": model}
            if image_b64:
                kwargs["image_b64"] = image_b64
            return module.chat(**kwargs)
    raise ValueError(f"Unknown model: {model}. Must contain one of: {', '.join(MODEL_MAP)}")

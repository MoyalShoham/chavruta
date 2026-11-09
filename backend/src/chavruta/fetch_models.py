"""Downloads model weights into the Hugging Face cache. Build time only.

This is the one step that talks to huggingface.co, and it runs outside the backend: in the Docker
build, or once on a developer machine. The backend itself runs with HF_HUB_OFFLINE=1 and the network
guard, so it can only load weights that are already on disk.

python -m chavruta.fetch_models [e5-small bge-m3 ...] [--chosen] [--reranker]
"""

from __future__ import annotations

import argparse

from chavruta.embeddings import CHOSEN_MODEL, MODELS, RERANKER


def wanted(argv: list[str] | None = None) -> list[tuple[str, str]]:
    """(Hugging Face name, weight file pattern) for each model asked for."""
    parser = argparse.ArgumentParser(prog="python -m chavruta.fetch_models")
    # No `choices` here: argparse validates an empty default list against them and fails.
    parser.add_argument("models", nargs="*", help=f"any of {', '.join(sorted(MODELS))}")
    parser.add_argument("--chosen", action="store_true", help=f"the model the app uses ({CHOSEN_MODEL})")
    parser.add_argument("--reranker", action="store_true")
    args = parser.parse_args(argv)
    unknown = [m for m in args.models if m not in MODELS]
    if unknown:
        parser.error(f"unknown models: {', '.join(unknown)}")
    keys = [*args.models, *([CHOSEN_MODEL] if args.chosen else [])]
    out = [(MODELS[k].name, MODELS[k].weights) for k in dict.fromkeys(keys)]
    if args.reranker:
        out.append((RERANKER, "*.safetensors"))
    return out


def main(argv: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    for name, weights in wanted(argv):
        # One weight format per model; no ONNX/OpenVINO duplicates.
        path = snapshot_download(
            name,
            allow_patterns=["*.json", weights, "*.txt", "*.model", "sentencepiece*", "1_Pooling/*"],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()

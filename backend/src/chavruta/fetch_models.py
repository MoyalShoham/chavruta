"""Downloads model weights into the Hugging Face cache. Build time only.

This is the one step that talks to huggingface.co, and it runs outside the backend: in the Docker
build, or once on a developer machine. The backend itself runs with HF_HUB_OFFLINE=1 and the network
guard, so it can only load weights that are already on disk.

python -m chavruta.fetch_models [e5-small bge-m3 ...] [--chosen] [--reranker]
"""

from __future__ import annotations

import argparse

from chavruta.embeddings import CHOSEN_MODEL, MODELS, RERANKER


def main(argv: list[str] | None = None) -> None:
    from huggingface_hub import snapshot_download

    parser = argparse.ArgumentParser(prog="python -m chavruta.fetch_models")
    parser.add_argument("models", nargs="*", choices=sorted(MODELS), default=[])
    parser.add_argument("--chosen", action="store_true", help=f"the model the app uses ({CHOSEN_MODEL})")
    parser.add_argument("--reranker", action="store_true")
    args = parser.parse_args(argv)
    keys = [*args.models, *([CHOSEN_MODEL] if args.chosen else [])]
    wanted = [(MODELS[k].name, MODELS[k].weights) for k in dict.fromkeys(keys)]
    if args.reranker:
        wanted.append((RERANKER, "*.safetensors"))
    for name, weights in wanted:
        # One weight format per model; no ONNX/OpenVINO duplicates.
        path = snapshot_download(
            name,
            allow_patterns=["*.json", weights, "*.txt", "*.model", "sentencepiece*", "1_Pooling/*"],
            ignore_patterns=["onnx/*", "openvino/*"],
        )
        print(f"{name}: {path}", flush=True)


if __name__ == "__main__":
    main()

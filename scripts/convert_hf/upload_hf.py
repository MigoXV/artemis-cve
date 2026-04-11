from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import HfApi

DEFAULT_ROOT = Path("model-bin/MigoXV")
DEFAULT_REPOS = ("yoloe26-x-seg", "mobileclip2-b")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Upload local MigoXV model folders to Hugging Face Hub.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT))
    parser.add_argument("--namespace", default="MigoXV")
    parser.add_argument("--token", default=None)
    parser.add_argument("--private", action="store_true")
    parser.add_argument(
        "--repos",
        nargs="+",
        default=list(DEFAULT_REPOS),
        help="Subdirectories under --root to upload.",
    )
    return parser.parse_args()


def ensure_required_files(local_dir: Path) -> None:
    required = ("README.md", "config.json", "model.safetensors")
    missing = [name for name in required if not (local_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{local_dir} is missing required files: {', '.join(missing)}")


def main() -> None:
    args = parse_args()
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"root directory does not exist: {root}")

    api = HfApi(token=args.token)
    user = api.whoami()
    print(f"authenticated_as: {user.get('name') or user.get('fullname') or '<unknown>'}")

    for repo_name in args.repos:
        local_dir = root / repo_name
        if not local_dir.is_dir():
            raise FileNotFoundError(f"missing local repo directory: {local_dir}")
        ensure_required_files(local_dir)

        repo_id = f"{args.namespace}/{repo_name}"
        api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=bool(args.private),
            exist_ok=True,
        )
        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(local_dir),
            commit_message=f"Upload {repo_name} from local converted safetensors",
        )
        print(f"uploaded: {repo_id}")
        print(f"commit: {commit_info.oid}")
        print(f"url: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()

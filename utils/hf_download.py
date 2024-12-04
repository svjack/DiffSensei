import argparse

from huggingface_hub import snapshot_download, login


def main(args):
    login(token="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")
    snapshot_download(
        repo_id="jianzongwu/DiffSensei",
        ignore_patterns=[".gitattributes", "README.md"],
        local_dir="/data00/wjz/checkpoints/diffusion/diffsensei",
    )

    print("The End")


if __name__ == '__main__':
    """
    HF_ENDPOINT=https://hf-mirror.com \
    nohup python -m utils.hf_download \
        > nohup/download.out 2>&1 &
    """
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    main(args)

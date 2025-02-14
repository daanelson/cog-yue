import os
import subprocess
import time
WEIGHTS_BASE_URL = "https://weights.replicate.delivery/default/yue/"


def download_weights(filename: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)

    if not os.path.exists(f"{dest_dir}/{filename}"):
        print(f"⏳ Downloading {filename} to {dest_dir}")
        start = time.time()
        subprocess.check_call(
            [
                "pget",
                "--log-level",
                "warn",
                "-xf",
                f"{WEIGHTS_BASE_URL}/{filename}.tar",
                dest_dir,
            ],
            close_fds=False,
        )
        print(f"✅ Download completed in {time.time() - start:.2f} seconds")
    else:
        print(f"✅ {filename} already exists in {dest_dir}")

if __name__ == '__main__':
    download_weights("xcodec_mini_infer", "./inference")

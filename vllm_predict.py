import importlib
import os

from inference.vllm_yue_engine import VLLMYue
os.environ['HF_HUB_CACHE'] = './inference/models'
import tempfile
import subprocess
import shutil
import time
from typing import List

from cog import BasePredictor, Input, Path
from vllm import LLM, SamplingParams
from vllm.inputs.data import TokensPrompt

WEIGHTS_BASE_URL = "https://weights.replicate.delivery/default/yue/"

# just here for debuggin
class Timer:
    def __init__(self):
        self.start_time = time.time()
        self.last_time = time.time()
    
    def time(self, message):
        current_time = time.time()
        elapsed = current_time - self.last_time
        print(f"{message}: {elapsed:.3f} seconds")
        self.last_time = current_time

    def end(self):
        elapsed = time.time() - self.start_time
        print(f"total time: {elapsed:.3f} seconds")


class Predictor(BasePredictor):
    def download_weights(self, filename: str, dest_dir: str):
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

    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        stage1_model = "m-a-p/YuE-s1-7B-anneal-en-cot"
        stage2_model = "m-a-p/YuE-s2-1B-general"
        cog_version = importlib.metadata.version("cog")
        timer = Timer()
        print(f"Cog version: {cog_version}\n")

        self.download_weights(
            "models--m-a-p--YuE-s1-7B-anneal-en-cot", "./inference/models"
        )
        self.download_weights(
            "models--m-a-p--YuE-s2-1B-general", "./inference/models"
        )
        timer.time(f"Loading models")
        self.model = VLLMYue()
        timer.time(f"models loaded")

    def predict(
        self,
        genre_description: str = Input(
            description="Text containing genre tags that describe the musical style (e.g. instrumental, genre, mood, vocal timbre, vocal gender)",
            default="inspiring female uplifting pop airy vocal electronic bright vocal vocal",
        ),
        lyrics: str = Input(
            description="Lyrics for music generation. Must be structured in segments with [verse], [chorus], [bridge], or [outro] tags",
            default="[verse]\nOh yeah, oh yeah, oh yeah\n\n[chorus]\nOh yeah, oh yeah, oh yeah",
        ),
        num_segments: int = Input( # TODO: probably just generate the number of segments provided but max out at 10?
            description="Number of segments to generate", default=2, ge=1, le=10
        ),
        max_new_tokens: int = Input(
            description="Maximum number of new tokens to generate",
            default=1500,
            ge=500,
            le=3000,
        ),
        seed: int = Input(
            description="Set a seed for reproducibility. Random by default.",
            default=None,
        ),
    ) -> List[Path]:
        """Run YuE inference on the provided inputs"""

        seed = self.seed_or_random_seed(seed)

        # Validate inputs
        if not lyrics.strip():
            raise ValueError("Lyrics cannot be empty")

        if not any(
            tag in lyrics.lower()
            for tag in ["[verse]", "[chorus]", "[bridge]", "[outro]"]
        ):
            raise ValueError(
                "Lyrics must contain at least one [verse], [chorus], [bridge], or [outro] tag"
            )

        if not genre_description.strip():
            raise ValueError("Genre description cannot be empty")

        # Setup output directory
        output_dir = f"./output/{time.time()}/"
        os.makedirs(output_dir, exist_ok=True)

        # Empty output directory
        for item in os.listdir(output_dir):
            path = os.path.join(output_dir, item)
            if os.path.isdir(path):
                shutil.rmtree(path)
            else:
                os.remove(path)

        self.model.main_generate(genre_description, lyrics, max_new_tokens, num_segments, seed, output_dir)

        # Find output files in vocoder/mix directory and rename to output_N.mp3
        mix_dir = os.path.join(output_dir, "vocoder", "mix")
        output_files = []
        if os.path.exists(mix_dir):
            mp3_files = [f for f in os.listdir(mix_dir) if f.endswith(".mp3")]
            for idx, file in enumerate(mp3_files):
                old_path = os.path.join(mix_dir, file)
                new_name = (
                    "output.mp3" if len(mp3_files) == 1 else f"output_{idx+1}.mp3"
                )
                new_path = f"./{new_name}"
                os.rename(old_path, new_path)
                output_files.append(Path(new_path))

        return output_files

    def seed_or_random_seed(self, seed: int | None) -> int:
        # Max seed is 2147483647
        if not seed or seed <= 0:
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

        print(f"Using seed: {seed}\n")
        return seed

# generating tokens for segment 1 -> stage 1 generation complete - 45 sec
# processing frames 0 - 300 - 45 seconds
# so that's basically where it all is
import importlib
import os
os.environ['HF_HUB_CACHE'] = './models'
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
            "models--m-a-p--YuE-s1-7B-anneal-en-cot", "./models"
        )
        self.download_weights("xcodec_mini_infer", "./inference")
        self.download_weights(
            "models--m-a-p--YuE-s3-1B-general", "./models"
        )
        timer.time(f"Loading models")
        self.stage1_model = LLM(stage1_model, skip_tokenizer_init=True, dtype='bfloat16')

        self.stage2_model = LLM(stage2_model, skip_tokenizer_init=True, dtype='bfloat16')
        timer.time(f"models loaded")

        # TODO: so much more stuff oh my gosh. 

    def stage2_generate(model, prompt, batch_size=16):
        timer.time(f"Stage 2 generation with batch size {batch_size}")
        codec_ids = codectool.unflatten(prompt, n_quantizer=1)
        codec_ids = codectool.offset_tok_ids(
                        codec_ids,
                        global_offset=codectool.global_offset,
                        codebook_size=codectool.codebook_size,
                        num_codebooks=codectool.num_codebooks,
                    ).astype(np.int32)

        # Prepare prompt_ids based on batch size or single input
        if batch_size > 1:
            codec_list = []
            for i in range(batch_size):
                idx_begin = i * 300
                idx_end = (i + 1) * 300
                codec_list.append(codec_ids[:, idx_begin:idx_end])

            codec_ids = np.concatenate(codec_list, axis=0)
            prompt_ids = np.concatenate(
                [
                    np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
                    codec_ids,
                    np.tile([mmtokenizer.stage_2], (batch_size, 1)),
                ],
                axis=1
            )
        else:
            prompt_ids = np.concatenate([
                np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
                codec_ids.flatten(),  # Flatten the 2D array to 1D
                np.array([mmtokenizer.stage_2])
            ]).astype(np.int32)
            prompt_ids = prompt_ids[np.newaxis, ...]

        codec_ids = torch.as_tensor(codec_ids).to(device)
        prompt_ids = torch.as_tensor(prompt_ids).to(device)
        len_prompt = prompt_ids.shape[-1]

        block_list = LogitsProcessorList([BlockTokenRangeProcessor(0, 46358), BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)])

        # Teacher forcing generate loop
        timer.time("Starting teacher forcing generation loop...")
        for frames_idx in range(codec_ids.shape[1]):
            if frames_idx % 100 == 0:
                timer.time(f"Processing frame {frames_idx}/{codec_ids.shape[1]}")
            cb0 = codec_ids[:, frames_idx:frames_idx+1]
            prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
            input_ids = prompt_ids

            with torch.no_grad():
                stage2_output = model.generate(input_ids=input_ids,
                    min_new_tokens=7,
                    max_new_tokens=7,
                    eos_token_id=mmtokenizer.eoa,
                    pad_token_id=mmtokenizer.eoa,
                    logits_processor=block_list,
                )

            assert stage2_output.shape[1] - prompt_ids.shape[1] == 7, f"output new tokens={stage2_output.shape[1]-prompt_ids.shape[1]}"
            prompt_ids = stage2_output

        # Return output based on batch size
        if batch_size > 1:
            output = prompt_ids.cpu().numpy()[:, len_prompt:]
            output_list = [output[i] for i in range(batch_size)]
            output = np.concatenate(output_list, axis=0)
        else:
            output = prompt_ids[0].cpu().numpy()[len_prompt:]

        return output

def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
    timer.time(f"Starting Stage 2 inference with batch size {batch_size}")
    stage2_result = []
    for i in tqdm(range(len(stage1_output_set))):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[i]))

        if os.path.exists(output_filename):
            timer.time(f'{output_filename} stage2 has done.')
            continue

        timer.time(f"Processing file {i+1}/{len(stage1_output_set)}: {stage1_output_set[i]}")
        # Load the prompt
        prompt = np.load(stage1_output_set[i]).astype(np.int32)

        # Only accept 6s segments
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6

        timer.time(f"Output duration: {output_duration}s, Number of batches: {num_batch}")

        if num_batch <= batch_size:
            # If num_batch is less than or equal to batch_size, we can infer the entire prompt at once
            output = stage2_generate(model, prompt[:, :output_duration*50], batch_size=num_batch)
        else:
            # If num_batch is greater than batch_size, process in chunks of batch_size
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)
            timer.time(f"Processing in {num_segments} segments")

            for seg in range(num_segments):
                timer.time(f"Processing segment {seg+1}/{num_segments}")
                start_idx = seg * batch_size * 300
                # Ensure the end_idx does not exceed the available length
                end_idx = min((seg + 1) * batch_size * 300, output_duration*50)  # Adjust the last segment
                current_batch_size = batch_size if seg != num_segments-1 or num_batch % batch_size == 0 else num_batch % batch_size
                segment = stage2_generate(
                    model,
                    prompt[:, start_idx:end_idx],
                    batch_size=current_batch_size
                )
                segments.append(segment)

            # Concatenate all the segments
            output = np.concatenate(segments, axis=0)

        # Process the ending part of the prompt
        if output_duration*50 != prompt.shape[-1]:
            timer.time("Processing ending segment...")
            ending = stage2_generate(model, prompt[:, output_duration*50:], batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        output = codectool_stage2.ids2npy(output)

        # Fix invalid codes (a dirty solution, which may harm the quality of audio)
        # We are trying to find better one
        timer.time("Fixing invalid codes...")
        fixed_output = copy.deepcopy(output)
        for i, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequant = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[i, j] = most_frequant
        # save output
        timer.time(f"Saving Stage 2 output to {output_filename}")
        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)
    return stage2_result

stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=args.stage2_batch_size)


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
        num_segments: int = Input(
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

        # Create temporary files for genre and lyrics
        def create_temp_file(content: str, prefix: str) -> str:
            temp_file = tempfile.NamedTemporaryFile(
                delete=False, mode="w", prefix=prefix, suffix=".txt"
            )
            content = content.strip() + "\n\n"
            content = content.replace("\r\n", "\n").replace("\r", "\n")
            temp_file.write(content)
            temp_file.close()
            return temp_file.name

        genre_file = create_temp_file(genre_description, "genre_")
        lyrics_file = create_temp_file(lyrics, "lyrics_")

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

        try:
            # Change to inference directory
            os.chdir("./inference")

            # Run inference
            command = [
                "python",
                "infer.py",
                "--stage1_model",
                "m-a-p/YuE-s1-7B-anneal-en-cot",
                "--stage2_model",
                "m-a-p/YuE-s2-1B-general",
                "--genre_txt",
                genre_file,
                "--lyrics_txt",
                lyrics_file,
                "--run_n_segments",
                str(num_segments),
                "--stage2_batch_size",
                "16",
                "--output_dir",
                output_dir,
                "--cuda_idx",
                "0",
                "--max_new_tokens",
                str(max_new_tokens),
                "--seed",
                str(seed),
            ]

            subprocess.run(command, check=True)

            # Change back to root directory
            os.chdir("..")

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
                    new_path = os.path.join(mix_dir, new_name)
                    os.rename(old_path, new_path)
                    output_files.append(Path(new_path))

            return output_files

        finally:
            # Cleanup temp files
            os.remove(genre_file)
            os.remove(lyrics_file)

    def seed_or_random_seed(self, seed: int | None) -> int:
        # Max seed is 2147483647
        if not seed or seed <= 0:
            seed = int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF

        print(f"Using seed: {seed}\n")
        return seed

# generating tokens for segment 1 -> stage 1 generation complete - 45 sec
# processing frames 0 - 300 - 45 seconds
# so that's basically where it all is
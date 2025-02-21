# Prediction interface for Cog ⚙️
import base64
import datetime
import subprocess
import os
import requests
import time
import torch
import re
import pandas as pd
import numpy as np

from cog import BasePredictor, BaseModel, Input, Path
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torchaudio
from faster_whisper.vad import VadOptions

# === Patch faster-whisper’s internal storage function to cast features to float16 ===
import faster_whisper.transcribe as ft
_orig_get_ctranslate2_storage = ft.get_ctranslate2_storage
def _patched_get_ctranslate2_storage(features):
    # Convert the input features to float16 before converting to CTranslate2 storage.
    return _orig_get_ctranslate2_storage(features.astype(np.float16))
ft.get_ctranslate2_storage = _patched_get_ctranslate2_storage
# === End patch ===

class Output(BaseModel):
    segments: list
    language: str = None
    num_speakers: int = None

class Predictor(BasePredictor):

    def setup(self):
        """Load the Whisper model into memory for efficient predictions."""
        model_path = "/src/kb-whisper-large-ct2-new"
        # Load the model using faster-whisper with FP16 compute type.
        self.model = WhisperModel(
            model_path,
            device="cuda",
            compute_type="float16",
        )
        # Override mel filters: force the feature extractor to use 128 mel bins.
        self.model.feature_extractor.mel_filters = self.model.feature_extractor.get_mel_filters(
            self.model.feature_extractor.sampling_rate,
            self.model.feature_extractor.n_fft,
            n_mels=128
        )
        # We no longer load the diarization pipeline here.
        # It will be loaded in predict() if enabled.

    def predict(
        self,
        enable_diarization: bool = Input(
            description="Perform speaker diarization? (Requires Hugging Face token)", default=True
        ),
        hf_token: str = Input(
            description="Your Hugging Face API token. Leave blank to use the default set in the environment.", default=None
        ),
        file_string: str = Input(
            description="Either provide: Base64 encoded audio file,", default=None
        ),
        file_url: str = Input(
            description="Or provide: A direct audio file URL", default=None
        ),
        file: Path = Input(
            description="Or an audio file", default=None
        ),
        num_speakers: int = Input(
            description="Number of speakers, leave empty to autodetect.", ge=1, le=50, default=None
        ),
        translate: bool = Input(
            description="Translate the speech into English.", default=False
        ),
        language: str = Input(
            description="Language of the spoken words as a language code like 'en'. Leave empty to auto-detect language.", default=None
        ),
        prompt: str = Input(
            description="Vocabulary: provide names, acronyms and loanwords in a list. Use punctuation for best accuracy.", default=None
        ),
    ) -> Output:
        """Run a single prediction on the model using faster-whisper."""
        
        # If diarization is enabled, try to load the diarization pipeline.
        if enable_diarization:
            # Use the token provided or fallback to the environment variable.
            hf_token = hf_token or os.getenv("HF_TOKEN")
            if hf_token:
                self.diarization_model = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token,
                ).to(torch.device("cuda"))
            else:
                print("Warning: Diarization enabled but no HF token provided; skipping diarization.")
                self.diarization_model = None
        else:
            self.diarization_model = None

        try:
            # Generate a temporary filename for the WAV file.
            temp_wav_filename = f"temp-{time.time_ns()}.wav"

            if file is not None:
                subprocess.run(
                    [
                        "ffmpeg", "-i", file, "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", temp_wav_filename,
                    ],
                    check=True,
                )
            elif file_url is not None:
                response = requests.get(file_url)
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as f:
                    f.write(response.content)
                subprocess.run(
                    [
                        "ffmpeg", "-i", temp_audio_filename, "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", temp_wav_filename,
                    ],
                    check=True,
                )
                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)
            elif file_string is not None:
                audio_data = base64.b64decode(
                    file_string.split(",")[1] if "," in file_string else file_string
                )
                temp_audio_filename = f"temp-{time.time_ns()}.audio"
                with open(temp_audio_filename, "wb") as f:
                    f.write(audio_data)
                subprocess.run(
                    [
                        "ffmpeg", "-i", temp_audio_filename, "-ar", "16000", "-ac", "1",
                        "-c:a", "pcm_s16le", temp_wav_filename,
                    ],
                    check=True,
                )
                if os.path.exists(temp_audio_filename):
                    os.remove(temp_audio_filename)

            segments, detected_num_speakers, detected_language = self.speech_to_text(
                temp_wav_filename,
                num_speakers,
                prompt,
                language,
                translate=translate,
            )
            print("Done with inference")
            return Output(
                segments=segments,
                language=detected_language,
                num_speakers=detected_num_speakers,
            )
        except Exception as e:
            raise RuntimeError("Error running inference with local model", e)
        finally:
            if os.path.exists(temp_wav_filename):
                os.remove(temp_wav_filename)

    def convert_time(self, secs, offset_seconds=0):
        return datetime.timedelta(seconds=(round(secs) + offset_seconds))

    def speech_to_text(
        self,
        audio_file_wav,
        num_speakers=None,
        prompt="",
        language=None,
        translate=False,
    ):
        time_start = time.time()
        print("Starting transcribing")
        options = dict(
            language=language,
            beam_size=5,
            vad_filter=True,
            vad_parameters=VadOptions(
                max_speech_duration_s=self.model.feature_extractor.chunk_length,
                min_speech_duration_ms=100,
                speech_pad_ms=100,
                threshold=0.25,
                neg_threshold=0.2,
            ),
            word_timestamps=True,
            initial_prompt=prompt,
            language_detection_segments=1,
            task="translate" if translate else "transcribe",
        )
        segments, transcript_info = self.model.transcribe(audio_file_wav, **options)
        segments = list(segments)
        segments = [
            {
                "avg_logprob": s.avg_logprob,
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text,
                "words": [
                    {
                        "start": float(w.start),
                        "end": float(w.end),
                        "word": w.word,
                        "probability": w.probability,
                    }
                    for w in s.words
                ],
            }
            for s in segments
        ]
        time_transcribing_end = time.time()
        print(f"Finished transcribing, took {time_transcribing_end - time_start:.5f} seconds, {len(segments)} segments")

        # Only perform diarization if the pipeline exists.
        if self.diarization_model is not None:
            print("Starting diarization")
            waveform, sample_rate = torchaudio.load(audio_file_wav)
            diarization = self.diarization_model(
                {"waveform": waveform, "sample_rate": sample_rate},
                num_speakers=num_speakers,
            )
            time_diarization_end = time.time()
            print(f"Finished with diarization, took {time_diarization_end - time_transcribing_end:.5f} seconds")

            print("Starting merging")
            diarize_segments = []
            diarization_list = list(diarization.itertracks(yield_label=True))
            for turn, _, speaker in diarization_list:
                diarize_segments.append({"start": turn.start, "end": turn.end, "speaker": speaker})
            diarize_df = pd.DataFrame(diarize_segments)
            unique_speakers = {speaker for _, _, speaker in diarization_list}
            detected_num_speakers = len(unique_speakers)

            final_segments = []
            for segment in segments:
                diarize_df["intersection"] = np.minimum(diarize_df["end"], segment["end"]) - np.maximum(diarize_df["start"], segment["start"])
                diarize_df["union"] = np.maximum(diarize_df["end"], segment["end"]) - np.minimum(diarize_df["start"], segment["start"])
                dia_tmp = diarize_df[diarize_df["intersection"] > 0]
                if len(dia_tmp) > 0:
                    speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                else:
                    speaker = "UNKNOWN"
                words_with_speakers = []
                for word in segment["words"]:
                    diarize_df["intersection"] = np.minimum(diarize_df["end"], word["end"]) - np.maximum(diarize_df["start"], word["start"])
                    diarize_df["union"] = np.maximum(diarize_df["end"], word["end"]) - np.minimum(diarize_df["start"], word["start"])
                    dia_tmp = diarize_df[diarize_df["intersection"] > 0]
                    if len(dia_tmp) > 0:
                        word_speaker = dia_tmp.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                    else:
                        word_speaker = speaker
                    word["speaker"] = word_speaker
                    words_with_speakers.append(word)
                new_segment = {
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": speaker,
                    "avg_logprob": segment["avg_logprob"],
                    "words": words_with_speakers,
                }
                final_segments.append(new_segment)

            if len(final_segments) > 0:
                grouped_segments = []
                current_group = final_segments[0].copy()
                sentence_end_pattern = r"[.!?]+"
                for segment in final_segments[1:]:
                    time_gap = segment["start"] - current_group["end"]
                    current_duration = current_group["end"] - current_group["start"]
                    can_combine = (
                        segment["speaker"] == current_group["speaker"]
                        and time_gap <= 1.0
                        and current_duration < 30.0
                        and not re.search(sentence_end_pattern, current_group["text"][-1:])
                    )
                    if can_combine:
                        current_group["end"] = segment["end"]
                        current_group["text"] += " " + segment["text"]
                        current_group["words"].extend(segment["words"])
                    else:
                        grouped_segments.append(current_group)
                        current_group = segment.copy()
                grouped_segments.append(current_group)
                final_segments = grouped_segments

            for segment in final_segments:
                segment["text"] = re.sub(r"\s+", " ", segment["text"]).strip()
                segment["text"] = re.sub(r"\s+([.,!?])", r"\1", segment["text"])
                segment["duration"] = segment["end"] - segment["start"]

            time_merging_end = time.time()
            print(f"Finished with merging, took {time_merging_end - time_diarization_end:.5f} seconds")
            return final_segments, detected_num_speakers, transcript_info.language
        else:
            print("Skipping diarization step.")
            # Return the transcription segments without diarization info.
            return segments, None, transcript_info.language

"""
================================================================================
VOICE TRAINING AND CHECKPOINT GENERATION
================================================================================
This script trains the TTS model's voice conditionals from a reference audio
file and saves them as a reusable checkpoint.

This process needs to be run only ONCE before using main.py.
After this, main.py can load the checkpoint instantly without reprocessing
the reference audio, making subsequent generation much faster.

Workflow:
1. Load the Chatterbox multilingual TTS model
2. Extract voice characteristics from reference audio (gai.wav)
3. Save voice conditionals to a checkpoint file
4. Use checkpoint in main.py for fast voice loading
================================================================================
"""

import torch
import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from pathlib import Path

# ============================================================================
# STEP 1: Load the Pre-trained TTS Model
# ============================================================================

# Load the Chatterbox multilingual TTS model from HuggingFace
# This model includes:
#   - T3: Text-to-token model (converts text to speech tokens)
#   - S3Gen: Token-to-audio model (converts tokens to waveform)
#   - Voice encoder: Extracts speaker characteristics
print("🔄 Loading Chatterbox Multilingual TTS model...")
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
print("✅ Model loaded successfully!\n")

# ============================================================================
# STEP 2: Configure Audio Paths
# ============================================================================

# Path to the reference audio file used for voice extraction
# This audio should be clear speech in the target language (Italian)
# Recommended: 5-30 seconds of clean speech
AUDIO_PROMPT_PATH = "train.wav"

# Output path where the voice checkpoint will be saved
# This checkpoint will be loaded by main.py
VOICE_CHECKPOINT = "voice_checkpoint.pt"

# ============================================================================
# STEP 3: Extract Voice Characteristics from Reference Audio
# ============================================================================

# Prepare voice conditionals from the reference audio
# This process:
#   - Loads the audio file from disk
#   - Extracts speaker embeddings using the voice encoder
#   - Computes F0 predictions (fundamental frequency)
#   - Prepares all acoustic features needed for generation
# The exaggeration parameter (0.5) controls emotional intensity
#   - 0.0 = neutral, 0.5 = moderate, 1.0+ = highly expressive
print(f"🎙️ Training voice from {AUDIO_PROMPT_PATH}...")
multilingual_model.prepare_conditionals(AUDIO_PROMPT_PATH, exaggeration=0.5)
print("✅ Voice trained!")

# ============================================================================
# STEP 4: Save Voice Checkpoint for Reuse
# ============================================================================

# Save the trained voice conditionals to disk
# The checkpoint contains all extracted voice characteristics
# This allows main.py to load the voice instantly without reprocessing
print(f"💾 Saving voice checkpoint to {VOICE_CHECKPOINT}...")
torch.save(multilingual_model.conds, VOICE_CHECKPOINT)
print("✅ Voice checkpoint saved!\n")

# ============================================================================
# SUCCESS MESSAGE
# ============================================================================

print(f"📝 Now you can use 'python main.py' to generate audio with your voice!")

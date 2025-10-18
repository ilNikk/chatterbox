"""
================================================================================
INTERACTIVE TEXT-TO-SPEECH GENERATOR WITH PRE-TRAINED VOICE
================================================================================
This script provides an interactive interface for generating Italian speech
using the Chatterbox multilingual TTS model with a pre-trained voice checkpoint.

Features:
- Loads the model and voice once at startup (efficient)
- Runs in an interactive loop for continuous generation
- Saves each output with an incremental counter
- Supports graceful exit and error handling
================================================================================
"""

import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from pathlib import Path
import sys
import torch

# ============================================================================
# INITIALIZATION: Load Model and Voice Checkpoint
# ============================================================================

# Load the pre-trained Chatterbox multilingual TTS model from HuggingFace
# This model supports 23 languages including Italian
# Loaded on CUDA GPU for fast inference
print("🔄 Loading Chatterbox Multilingual TTS model...")
multilingual_model = ChatterboxMultilingualTTS.from_pretrained(device="cuda")
print("✅ Model loaded successfully!")

# Load the pre-trained voice conditionals from a checkpoint file
# The checkpoint contains:
#   - Speaker embeddings extracted from the reference audio
#   - Acoustic characteristics and voice identity
#   - F0 predictions and other voice parameters
# This was created by train_voice.py from gai.wav
VOICE_CHECKPOINT = "voice_checkpoint.pt"
if Path(VOICE_CHECKPOINT).exists():
    print(f"🎙️ Loading voice from {VOICE_CHECKPOINT}...")
    # Use weights_only=False to support custom PyTorch classes (Conditionals)
    multilingual_model.conds = torch.load(
        VOICE_CHECKPOINT, 
        map_location="cuda", 
        weights_only=False
    )
    print("✅ Voice loaded successfully!\n")
else:
    print(f"⚠️ Voice checkpoint not found!")
    print(f"📝 Run 'python train_voice.py' first to create it.\n")
    sys.exit(1)

# ============================================================================
# CONFIGURATION: Generation Parameters
# ============================================================================

# Counter to track the number of generated audio files
counter = 1

# Generation parameters optimized for fast inference with good quality
# Temperature: Controls diversity of generated tokens
#   - Lower values (0.6) = more deterministic, faster generation
#   - Higher values = more random variations in speech
TEMPERATURE = 0.6

# Repetition penalty: Prevents the model from repeating the same tokens
#   - Ensures more natural, varied speech output
REPETITION_PENALTY = 1.5

# ============================================================================
# MAIN LOOP: Interactive Speech Generation
# ============================================================================

while True:
    try:
        # Prompt user for input text to synthesize
        text = input("📝 Enter text (or 'quit' to exit): ").strip()
        
        # Handle exit command
        if text.lower() == 'quit':
            print("👋 Goodbye!")
            break
        
        # Validate that user entered some text
        if not text:
            print("⚠️ Please enter some text.\n")
            continue
        
        # Generate Italian speech from the input text
        # Uses the pre-trained voice conditionals loaded at startup
        print(f"🎤 Generating audio in Italian...")
        wav = multilingual_model.generate(
            text,                      # Input text to synthesize
            language_id="it",          # Italian language
            temperature=TEMPERATURE,   # Generation randomness parameter
            repetition_penalty=REPETITION_PENALTY  # Avoid token repetition
        )
        
        # Save the generated audio waveform to a WAV file
        # Files are saved with incremental names (output_1.wav, output_2.wav, etc.)
        output_file = f"output_{counter}.wav"
        ta.save(output_file, wav, multilingual_model.sr)
        print(f"✅ Saved to: {output_file}\n")
        counter += 1
        
    except KeyboardInterrupt:
        # Handle Ctrl+C gracefully without error messages
        print("\n\n👋 Goodbye!")
        break
    except Exception as e:
        # Catch and display errors without crashing the loop
        # Allows user to recover and continue generating audio
        print(f"❌ Error: {e}\n")
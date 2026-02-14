# Test Audio Files

Place audio files in this directory for testing the audio analysis features.

## Supported Formats
- WAV (.wav)
- MP3 (.mp3)
- FLAC (.flac)
- AIFF (.aiff)
- OGG (.ogg)

## How to Use

1. Copy audio files to this directory
2. Build and run the application
3. In Debug mode, select "File Playback" as input source
4. Select the audio file from the dropdown
5. Click Play to start analysis

## Notes

- Audio files are loaded at runtime (not embedded in executable)
- Files are automatically discovered on startup
- Recommended: Use short clips (< 30 seconds) for quick testing
- Files are played in loop mode by default

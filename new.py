import os
import google.generativeai as genai
import whisperx
from pydub import AudioSegment

# ✅ Step 1: Read Audio File
def read_audio(audio_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        print(f"Duration: {len(audio) / 1000} seconds, Channels: {audio.channels}")
        return audio
    except Exception as e:
        print(f"Error reading audio file: {e}")
        return None

# ✅ Step 2: Transcribe Audio Using WhisperX
def transcribe_audio(audio_path):
    try:
        model = whisperx.load_model("base", device="cpu", compute_type="float32")  # Ensure CPU compatibility
        result = model.transcribe(audio_path)

        print("🔍 DEBUG: Full WhisperX Output")
        print(result)  # Print the entire output to check its structure

        # Check if 'segments' exist and extract text
        if "segments" in result and isinstance(result["segments"], list):
            lyrics_list = [seg["text"] for seg in result["segments"] if "text" in seg]
            lyrics = " ".join(lyrics_list).strip()

            if lyrics:
                return lyrics
            else:
                return "❌ No lyrics found in segments!"
        else:
            return "❌ No valid segments found in WhisperX output!"
    except Exception as e:
        return f"❌ Error in transcription: {e}"


# ✅ Step 3: Identify Song Using Gemini
def identify_song_with_gemini(lyrics, model):
    query = f"What is the song name and artist for these lyrics: {lyrics}?"
    try:
        response = model.generate_content(query)
        return response.text.strip()
    except Exception as e:
        return f"Error querying Gemini: {e}"

# ✅ Step 4: Check Copyright with Gemini
def copyright_check_with_gemini(song_name, artist, model):
    query = f"Is the song '{song_name}' by {artist} copyrighted?"
    try:
        response = model.generate_content(query)
        return response.text.strip()
    except Exception as e:
        return f"Error querying Gemini: {e}"

# ✅ Step 5: Main Function
if __name__ == "__main__":
    audio_path = "D:\\fold\\Project\\acd\\sample.mp3"
    audio = read_audio(audio_path)
    
    if audio:
        print("\n🔹 Transcribing audio...")
        lyrics = transcribe_audio(audio_path)
        print(f"📝 Transcribed Lyrics: {lyrics}")

        # ✅ Configure Gemini AI
        API_KEY = "AIzaSyCQR_jGzc24rDKLeXgEGc5CblyGTY4aH38"  
        genai.configure(api_key=API_KEY)
        model = genai.GenerativeModel("gemini-1.5-pro")

        print("\n🔍 Identifying song using Gemini...")
        song_info = identify_song_with_gemini(lyrics, model)
        print(f"🎵 Song Info from Gemini: {song_info}")

        # Extract song name and artist if the response is valid
        if "by" in song_info:
            parts = song_info.split("by")
            if len(parts) >= 2:
                song_name = parts[0].strip()
                artist = parts[1].strip()

                print("\n🔎 Checking for copyright...")
                copyright_info = copyright_check_with_gemini(song_name, artist, model)
                print(f"⚖️ Copyright Info: {copyright_info}")
            else:
                print("❌ Could not properly extract song details.")
        else:
            print("❌ Gemini response did not contain song details in expected format.")

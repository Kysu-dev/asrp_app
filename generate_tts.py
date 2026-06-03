import os
import asyncio
import edge_tts

DIGITS = {
    "0": "Nol",
    "1": "Satu",
    "2": "Dua",
    "3": "Tiga",
    "4": "Empat",
    "5": "Lima",
    "6": "Enam",
    "7": "Tujuh",
    "8": "Delapan",
    "9": "Sembilan"
}

VOICE_MALE = "id-ID-ArdiNeural"
VOICE_FEMALE = "id-ID-GadisNeural"

OUTPUT_DIR = "static/audio"

async def generate():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for digit, text in DIGITS.items():
        print(f"Generating {digit}...")
        
        # Male
        communicate = edge_tts.Communicate(text, VOICE_MALE)
        await communicate.save(os.path.join(OUTPUT_DIR, f"{digit}_male.mp3"))
        
        # Female
        communicate = edge_tts.Communicate(text, VOICE_FEMALE)
        await communicate.save(os.path.join(OUTPUT_DIR, f"{digit}_female.mp3"))

if __name__ == "__main__":
    asyncio.run(generate())
    print("Done generating TTS audio!")

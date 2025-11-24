from langdetect import detect

text = "Das ist ein Test."
try:
    lang = detect(text)
    print(f"Text: {text}")
    print(f"Lang: {lang}")
except Exception as e:
    print(f"Error: {e}")

text_en = "This is a test."
try:
    lang = detect(text_en)
    print(f"Text: {text_en}")
    print(f"Lang: {lang}")
except Exception as e:
    print(f"Error: {e}")

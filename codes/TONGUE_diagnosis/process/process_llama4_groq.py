import os
import json
import time
import base64
from groq import Groq

# Read Groq API key from environment variable
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Please set GROQ_API_KEY environment variable first!")

client = Groq(api_key=GROQ_API_KEY)

INPUT_PATH = '/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/val.sharegpt.json'
OUTPUT_PATH = '/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/val.llama4groq_output.jsonl'
IMG_BASE = '/home/joongwon00/Project_Tsinghua_Paper/TCMPipe/dataset/'
PROMPT_TEXT = (
    "You are an expert in Chinese medicine tongue diagnosis. "
    "Please analyze the tongue photos provided based on Chinese medicine tongue diagnosis and output the results in a single sentence, as in the example:"
    "(舌尖红苔薄腻, 舌淡红苔薄白, 舌胖苔薄白 etc)."
)
RATE_LIMIT_SECONDS = 1
MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"

# Encode image file to base64 data URL
def encode_image_to_data_url(image_path):
    ext = os.path.splitext(image_path)[1].lower()
    mime = 'image/jpeg' if ext in ['.jpg', '.jpeg'] else 'image/png'
    with open(image_path, "rb") as image_file:
        b64 = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:{mime};base64,{b64}"

# Send multimodal (image+text) prompt to Groq Llama-4-Scout (stream)
def llama4groq_infer(image_path: str, prompt: str) -> str:
    try:
        data_url = encode_image_to_data_url(image_path)
        content = [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": data_url}}
        ]
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            temperature=0.2,
            max_tokens=64,
            top_p=1,
            stream=True,
        )
        result = ""
        for chunk in completion:
            result += chunk.choices[0].delta.content or ""
        return result.strip()
    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        return "[API_ERROR]"

def main():
    with open(INPUT_PATH, 'r', encoding='utf-8') as f:
        data = json.load(f)
    results = []
    for idx, sample in enumerate(data, 1):
        rel = sample["images"][0] if "images" in sample and sample["images"] else ""
        abs_path = os.path.join(IMG_BASE, rel)
        if not os.path.exists(abs_path):
            result = "[FILE_NOT_FOUND]"
        else:
            result = llama4groq_infer(abs_path, PROMPT_TEXT)
        results.append({"image": rel, "llama4groq_output": result})
        print(f"[{idx}/{len(data)}] {rel} → {result}")
        time.sleep(RATE_LIMIT_SECONDS)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\n✅  Complete → {OUTPUT_PATH}")

if __name__ == "__main__":
    main() 
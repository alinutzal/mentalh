import json
import pandas as pd
import requests
#import tdqm

MODEL_NAME = "Qwen/Qwen3-8B"
VLLM_URL = "http://localhost:8000/v1/completions"
BATCH_SIZE = 64
MAX_CHARS = 1500

def build_prompt(post):
    post = str(post)
    post = post[:MAX_CHARS]    
    return f"""
Do not repeat the post. Do not add text.
Classify this Reddit post.

Topics:
- college: true if related to school, university, classes, exams, homework, GPA, campus, professors, students, college life, college stress, 
college mental health, college relationships, college social life, college events, college clubs, college sports, college housing, 
college finances, college applications, college admissions, college scholarships, college majors, college courses, college professors, 
college students, assignments, dorms, campus, tuition, academic stress, graduation, major, degree, studying.or any other topic related to higher education.
- gaming: true if related to video games, esports, online games, consoles, streaming games, gaming culture, gaming communities, 
gaming events, game development, gaming news, gaming trends, gaming industry, gaming hardware, gaming software, gaming accessories, 
gaming merchandise, gaming lifestyle, gaming humor, gaming memes, gaming challenges, gaming achievements, gaming discussions, gaming reviews, 
gaming recommendations, gaming tips and tricks, gaming tutorials, gaming guides, gaming walkthroughs, gaming let's plays, gaming live streams, 
gaming podcasts, gaming forums, Steam, Xbox, PlayStation, Nintendo, Valorant, League of Legends, Fortnite,Minecraft, Roblox, Call of Duty,
Counter Strike, CS2, Apex, gaming addiction, gaming friends, streaming, Twitch, Discord, ranked matches, MMORPG, RPG, FPS, spending significant time gaming.
- mental_health: true if the post is about mental health or contains clear signs of depression, anxiety, suicidal ideation, 
other mental health issuesstress, burnout, loneliness, isolation, depression, anxiety, panic, self-esteem, self-hatred, emotional distress,
mental illness, therapy, counseling, suicide, self-harm, or significant psychological suffering.

Return ONLY valid JSON with this exact format:
{{
  "mental_health": false,
  "college": false,
  "gaming": false,
}}

Do not break this down. Just return the JSON answer.

Post:
{post}

JSON:
""".strip()


def parse_response(text):
    text = text.strip()
    text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON found: {text}")

    decoder = json.JSONDecoder()
    data, _ = decoder.raw_decode(text[start:])

    return {
        "college": bool(data.get("college", False)),
        "gaming": bool(data.get("gaming", False)),
        "mental_health": bool(data.get("mental_health", False)),
    }


session = requests.Session()
df = pd.read_csv("data/mental_health_combined_test.csv")
df = pd.read_csv("data/mental_health_unbanlanced.csv")

results = []

for start in range(0, len(df), BATCH_SIZE):
    batch = df.iloc[start:start + BATCH_SIZE]
    prompts = [build_prompt(str(post)) for post in batch["text"]]

    payload = {
        "model": MODEL_NAME,
        "prompt": prompts,
        "temperature": 0.01,
        "top_p": 1.0,
        "max_tokens": 45
    }

    response = session.post(VLLM_URL, json=payload, timeout=300)
    response.raise_for_status()

    data = response.json()

    for choice in data["choices"]:
        try:
            results.append(parse_response(choice["text"]))
        except Exception as e:
            print("Parse failed:", repr(choice["text"]), e)
            results.append({
                "mental_health": False,
                "college": False,
                "gaming": False
            })

    print(f"Processed {min(start + BATCH_SIZE, len(df))}/{len(df)}")

labels_df = pd.DataFrame(results)
output = pd.concat([df.reset_index(drop=True), labels_df], axis=1)

output.to_csv("results/reddit_classified.csv", index=False)
print("Saved reddit_classified.csv")
from predict import Predictor

p = Predictor()
p.setup()
lyrics = """[verse]
Staring at the sunset, colors paint the sky
Thoughts of you keep swirling, can't deny
I know I let you down, I made mistakes
But I'm here to mend the heart I didn't break

[chorus]
Every road you take, I'll be one step behind
Every dream you chase, I'm reaching for the light
You can't fight this feeling now
I won't back down
You know you can't deny it now
I won't back down
"""
p.predict(        
    genre_description = "inspiring female uplifting pop airy vocal electronic bright vocal vocal",
    lyrics=lyrics,
    num_segments=2,
    max_new_tokens=1500,
    seed=1234,
)
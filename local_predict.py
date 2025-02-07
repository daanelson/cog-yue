from predict import Predictor

p = Predictor()
p.setup()
p.predict(        
    genre_description = "inspiring female uplifting pop airy vocal electronic bright vocal vocal",
    lyrics= "[verse]\nOh yeah, oh yeah, oh yeah\n\n[chorus]\nOh yeah, oh yeah, oh yeah",
    num_segments=2,
    max_new_tokens=1500,
    seed=1234,
)
from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
result=generator("The White man worked as a", max_length=10, num_return_sequences=5)
print(result[0]['generated_text'])

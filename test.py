from transformers import pipeline

qa = pipeline(
  "question-answering",
  model=r'C:\Users\tejas\Documents\Deep Learning\My Work\transformers\qa_model\checkpoint-23500',
  device=0,
)


context = "I want to study programming today."
question = "What do I want to study?"

print(qa(context=context, question=question))

from transformers import pipeline
import streamlit as st

st.title('Question Answering Transformer')

st.markdown('''This is a transformer used for question answering. Enter a question and the context 
            surrounding your question in the text boxes below.''')

st.markdown('''For example: 
                Question: "What did I buy today?"
                Context: "I went out of the house to buy some milk.''')

st.markdown('''The model will answer your question based on the context provided.''')


context = st.text_area("Context", "I went out of the house to buy some milk.")

question = st.text_input('Question', "What did I buy today?")

        
qa = pipeline(
  "question-answering",
  model=r'C:\Users\tejas\Documents\Deep Learning\My Work\transformers\qa_model\checkpoint-23500',
  device=0,
)

st.write(qa(context=context, question=question))
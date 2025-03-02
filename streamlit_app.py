import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "silanm/nlp-a5"


@st.cache_resource()  # Cache the model loading for performance
def load_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


model, tokenizer = load_model_and_tokenizer()

st.title("DPO Model Interaction")
st.write("Enter a prompt and the model will generate a response.")

prompt = st.text_area("Enter your prompt:", height=150)

if st.button("Generate Response"):
    if prompt:
        with st.spinner("Generating..."):
            inputs = tokenizer(prompt, return_tensors="pt")

            # --- Generate Response ---
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                pad_token_id=tokenizer.pad_token_id,
                no_repeat_ngram_size=3,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
            )

            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.write("### Response:")
            st.write(response)
    else:
        st.warning("Please enter a prompt.")

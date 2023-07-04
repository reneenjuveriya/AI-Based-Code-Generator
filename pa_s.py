import streamlit as st
from transformers import TFT5ForConditionalGeneration, RobertaTokenizer
from pathlib import Path

class Args:
    # Define training arguments
    
    # Model
    model_type = 't5'
    tokenizer_name = 'Salesforce/codet5-base'
    model_name_or_path = 'Salesforce/codet5-base'
    
    # Data
    train_batch_size = 8
    validation_batch_size = 8
    max_input_length = 48
    max_target_length = 128
    prefix = "Generate Python: "    

    # Optimizer
    learning_rate = 3e-4
    weight_decay = 1e-4
    warmup_ratio = 0.2
    adam_epsilon = 1e-8

    # Training
    seed = 2022
    epochs = 20

    # Directories
    output_dir = "runs/"
    logging_dir = f"{output_dir}/logs/"
    checkpoint_dir = f"checkpoint"
    save_dir = f"{output_dir}/saved_model/"
    cache_dir = '../working/'
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(logging_dir).mkdir(parents=True, exist_ok=True)
    Path(save_dir).mkdir(parents=True, exist_ok=True)

# Initialize training arguments
args = Args()

# Load the trained T5 model and tokenizer
model = TFT5ForConditionalGeneration.from_pretrained(args.save_dir)
tokenizer = RobertaTokenizer.from_pretrained(args.save_dir) 

def generate_code(query):
    # Encode texts by prepending the task for input sequence and appending the test sequence
    encoded_text = tokenizer(args.prefix + query, return_tensors='tf', padding='max_length', truncation=True, max_length=args.max_input_length)
    
    generated_code = model.generate(
        encoded_text["input_ids"], attention_mask=encoded_text["attention_mask"], 
        max_length=args.max_target_length, top_p=0.95, top_k=50, repetition_penalty=1.2, num_return_sequences=1)    
    # Decode generated tokens
    decoded_code = tokenizer.decode(generated_code.numpy()[0], skip_special_tokens=True)
    return decoded_code

# Define the Streamlit app
st.title('python Code Generation')

# Add a text input for the user to enter a query
query = st.text_input('Enter a natural language query:', '')

# Add a button to generate code from the query
if st.button('Generate code'):
    # Generate code from the query using the generate_code function
    generated_code = generate_code(query)
    
    # Display the generated code to the user
    st.code(generated_code, language='python')

import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch.nn.functional as F  # Import the softmax function

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
model = AutoModelForMaskedLM.from_pretrained("bert-base-chinese")

# Input and output file paths
input_file_path = "/Users/test.csv" #change to your own path
output_file_path = "/Users/out.csv"

# Open output file for appending
output_file = open(output_file_path, "a", encoding="utf-8")

# Open input file for reading
with open(input_file_path, "r", encoding="utf-8") as input_file:
    # Process each sentence in the input file
    for sentence in input_file:
        # Tokenize the sentence
        inputs = tokenizer(sentence.strip(), return_tensors="pt")

        # Generate predictions
        with torch.no_grad():
            outputs = model(**inputs)

        # Retrieve the predicted tokens and their probabilities
        masked_index = inputs['input_ids'][0].tolist().index(tokenizer.mask_token_id)
        logits = outputs.logits[0, masked_index]  # Get the logits for the masked position

        # Apply softmax to convert logits to probabilities
        probabilities = F.softmax(logits, dim=-1)

        # If you want to see the logits probabilities (before softmax), uncomment the following line
        #probabilities = logits

        # Get top 50 tokens and their probabilities
        topk_probabilities, topk_indices = torch.topk(probabilities, k=50)

        # Convert token indices to actual tokens using the tokenizer
        predicted_tokens = tokenizer.convert_ids_to_tokens(topk_indices.tolist())

        # Write the results to the output file
        output_file.write(f"Input sentence: {sentence.strip()}\n")
        for token, probability in zip(predicted_tokens, topk_probabilities):
            output_file.write(f"Word: {token}, Probability: {probability.item():.4f}\n")  # Use .item() to get Python float
        output_file.write("\n")

# Close the output file
output_file.close()

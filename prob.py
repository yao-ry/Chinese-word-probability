import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-bert-wwm")
model = AutoModelForMaskedLM.from_pretrained("hfl/chinese-bert-wwm")

# Input and output file paths
input_file_path = "/Users/runyi/Desktop/research idea/Mandarin studies/predication update/norming/test.csv"
output_file_path = "/Users/runyi/Desktop/research idea/Mandarin studies/predication update/norming/out.csv"

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
        topk_tokens = outputs.logits[0, masked_index].topk(50)
        predicted_tokens = tokenizer.convert_ids_to_tokens(topk_tokens.indices.tolist())
        predicted_probabilities = topk_tokens.values.tolist()

        # Write the results to the output file
        output_file.write(f"Input sentence: {sentence.strip()}\n")
        for token, probability in zip(predicted_tokens, predicted_probabilities):
            output_file.write(f"Word: {token}, Probability: {probability:.4f}\n")
        output_file.write("\n")

# Close the output file
output_file.close()

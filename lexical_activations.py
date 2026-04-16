# Imports
from transformers import BertModel, BertTokenizer, logging
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
import random
import re

# Suppress warning from unused pre-training heads
logging.set_verbosity_error()


# Load the BERT model and tokenizer
def load_model(model_name = "bert-base-uncased"):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states = True)
    model.eval()
    return tokenizer, model


# Function to remove a specific word from a sentence (case-insensitive, whole word match)
def remove_word(sentence, word):
    pattern = r'\b{}\b'.format(re.escape(word))
    sentence = re.sub(pattern, '', sentence, flags = re.IGNORECASE)
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence

# Function to create a modified DataFrame where the target word is removed from all sentences
def create_half_removed_df(df):

    # Copy dataframe
    df_mod = df.copy()

    # Loop through each row
    for idx, row in df_mod.iterrows():

        # Extract the target word and sentences
        word = str(row.iloc[0])
        sentences = row.iloc[1:].tolist()

        # Select sentence indices
        indices = list(range(len(sentences)))

        # Remove the target word from all sentences
        for i in indices:
            sentences[i] = remove_word(str(sentences[i]), word)

        # Update the DataFrame with modified sentences
        df_mod.iloc[idx, 1:] = sentences

    return df_mod


# Function to extract a single vector representation for a sentence by taking the max activation across tokens for each neuron
def extract_sentence_vector(sentence, tokenizer, model, layer_index = -1):

    # Get activations for the sentence
    inputs = tokenizer(sentence, return_tensors = "pt", truncation = True, max_length = 128)

    # Get hidden states for the specified layer
    with torch.no_grad():
        outputs = model(**inputs)
    hidden_state = outputs.hidden_states[layer_index].squeeze(0)

    # Exclude [CLS] and [SEP] tokens if present
    if hidden_state.shape[0] > 2:
        layer_acts = hidden_state[1:-1].numpy()
    else:
        layer_acts = hidden_state.numpy()

    # Take max activation across tokens for each neuron to get a single vector representation
    neuron_scores = layer_acts.max(axis = 0)

    return neuron_scores


# Function to determine the percentage of sentences that activate each neuron above the 95th percentile threshold, and return the top 10 neurons for each target word
def find_top_neurons(df, tokenizer, model):

    # Get number of neurons from model config
    num_neurons = model.config.hidden_size
    results = {}

    # Loop through each row in the DataFrame
    for idx, row in df.iterrows():

        # Get the target word and sentences
        word = str(row.iloc[0])
        sentences = row.iloc[1:].dropna().values

        # Array to hold counts of how many sentences activate each neuron above the threshold
        neuron_active_counts = np.zeros(num_neurons)

        # Loop through sentences
        for sent in sentences:
            
            # Extract neuron scores for the sentence
            neuron_scores = extract_sentence_vector(str(sent), tokenizer, model)

            # Convert to magnitudes and get the 95th percentile threshold
            magnitudes = np.abs(neuron_scores)
            threshold = np.percentile(magnitudes, 95)

            # Determine which neurons are active above the threshold
            is_active = (magnitudes >= threshold).astype(int)

            # Update counts for each neuron
            neuron_active_counts += is_active

        # Calculate the percentage of sentences that activate each neuron above the threshold
        percentages = neuron_active_counts / len(sentences) * 100

        # Get the top 10 neurons based on activation percentage
        top10 = np.argsort(percentages)[-10:][::-1]
        results[word] = top10

        print(f"Top neurons determined for '{word}'")

    return results


# Function to evaluate the activation of the identified neurons in both original and modified datasets, and return the percentage of sentences that activate each neuron above the threshold
def evaluate_neurons(df, tokenizer, model, neuron_map):

    # Dictionary to hold results for each word
    results = {}

    # Loop through each row in the DataFrame
    for idx, row in df.iterrows():

        # Get the target word, neurons to evaluate, and sentences
        word = str(row.iloc[0])
        neurons = neuron_map[word]
        sentences = row.iloc[1:].dropna().values

        # Array to hold counts of how many sentences activate each neuron above the threshold
        counts = np.zeros(len(neurons))

        # Loop through sentences
        for sent in sentences:

            # Extract neuron scores for the sentence
            neuron_scores = extract_sentence_vector(str(sent), tokenizer, model)

            # Convert to magnitudes and get the 95th percentile threshold
            magnitudes = np.abs(neuron_scores)
            threshold = np.percentile(magnitudes, 95)

            # Check if each of the identified neurons is active above the threshold and update counts
            for i, n in enumerate(neurons):
                if magnitudes[n] >= threshold:
                    counts[i] += 1

        # Calculate the percentage of sentences that activate each neuron above the threshold
        percentages = counts / len(sentences) * 100

        # Store results for the word
        results[word] = {
            "indices": neurons,
            "percentages": percentages
        }

    return results


# Function to plot the comparison of neuron activation between original and modified datasets
def plot_comparison(original_results, removed_results):

    # Get list of words from the results
    words = list(original_results.keys())

    # Initialize the figure and axes for subplots
    fig, axes = plt.subplots(len(words), 2, figsize = (16, 4*len(words)), sharey = True)

    # Loop through each target word
    for i, word in enumerate(words):

        # Loop through original and removed results for the word
        for j, (results, label) in enumerate([(original_results, "Present"), (removed_results, "Removed")]):

            # Get current axis for plotting
            ax = axes[i][j]

            # Get neuron indices and activation percentages for the current condition
            indices = results[word]["indices"]
            pcts = results[word]["percentages"]

            # Create x-axis labels based on neuron indices
            x_labels = [str(i) for i in indices]

            # Plot bar chart of activation percentages for the identified neurons
            sns.barplot(
                x = x_labels,
                y = pcts,
                ax = ax,
                palette = "viridis",
                hue = x_labels,
                legend = False
            )

            # Set y-axis limits and labels
            ax.set_ylim(0, 105)
            ax.set_xlabel("Neuron Index")

            if j == 0:
                ax.set_ylabel("Activation Frequency (%)")
            else:
                ax.set_ylabel("")

            # Set title for each subplot
            ax.set_title(f"{word} — {label}", fontweight = "bold")

            # Remove top and right spines for cleaner look
            sns.despine(ax = ax)

    # Add super title for the entire figure
    plt.suptitle(
        "Neuron Activation With vs Without Target Word\n(Top 10 Neurons from Sentences With Target Word Present)",
        fontsize = 14,
        fontweight = "bold",
        y = 1.01
    )

    # Show and save the plot
    plt.tight_layout()
    plt.savefig("lexical_comparison.png", dpi = 150, bbox_inches = "tight")
    plt.show()


# Main execution
if __name__ == "__main__":

    # Load in the dataset containing sentences with common words
    df = pd.read_csv("common_word_sentences_dataset.csv")

    # Load the BERT model and tokenizer
    tokenizer, model = load_model()

    # Identify top neurons from original data
    top_neurons = find_top_neurons(df, tokenizer, model)

    # Create modified dataset
    df_removed = create_half_removed_df(df)

    # Measure activation in both conditions
    original_results = evaluate_neurons(df, tokenizer, model, top_neurons)
    removed_results = evaluate_neurons(df_removed, tokenizer, model, top_neurons)

    # Plot the comparison of neuron activation between original and modified datasets
    plot_comparison(original_results, removed_results)
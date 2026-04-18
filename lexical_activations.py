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
def create_removed_df(df):

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

    # Scale figure width dynamically based on number of words
    fig, axes = plt.subplots(1, len(words), figsize = (5 * len(words), 7), sharey = True)

    # Handle the case where there is only one word (axes would not be a list)
    if len(words) == 1:
        axes = [axes]

    # Define color scheme for continuity (Teal for Present, Salmon for Removed)
    present_color = "steelblue"
    removed_color = "salmon"

    # Loop through each target word
    for i, word in enumerate(words):
        ax = axes[i]

        # Extract data for the current word
        indices = original_results[word]["indices"]
        present_pcts = original_results[word]["percentages"]
        removed_pcts = removed_results[word]["percentages"]

        # Compute Activation Drop Rate: mean percentage point drop across top 10 neurons
        max_drop = np.max(present_pcts - removed_pcts)

        # Create x-axis labels based on neuron indices (convert to string for categorical plotting)
        x_labels = [str(idx) for idx in indices]

        # Create a temporary DataFrame to make Seaborn plotting straightforward
        plot_df = pd.DataFrame({
            'Neuron Index': x_labels,
            'Present': present_pcts,
            'Removed': removed_pcts
        })

        # Plot Base Layer (Wider bars, Target Word Present)
        sns.barplot(
            x = 'Neuron Index',
            y = 'Present',
            data = plot_df,
            ax = ax,
            color = present_color,
            width = 0.85,
            edgecolor = present_color,
            alpha = 0.85,
            label = "Present"
        )

        # Plot Overlaid Layer (Narrower bars, Target Word Removed)
        sns.barplot(
            x = 'Neuron Index',
            y = 'Removed',
            data = plot_df,
            ax = ax,
            color = removed_color,
            width = 0.55,
            edgecolor = "white",   # White edge separates overlay from base bar
            alpha = 0.85,
            label = "Removed"
        )

        # Formatting & De-cluttering
        ax.set_ylim(0, 105)
        ax.set_xlabel("Neuron Index", fontsize = 12, fontweight = 'bold')
        ax.set_title(f"{word.upper()}", fontsize = 15, fontweight = "bold")
        ax.tick_params(axis = 'both', labelsize = 10)

        # Annotate with Activation Drop Rate just below the top of the subplot
        ax.text(
            x = 0.5,
            y = 0.995,
            s = f"Max Drop: {max_drop:.1f}%",
            transform = ax.transAxes,
            ha = 'center',
            va = 'top',
            fontsize = 11,
            fontweight = 'bold',
            color = 'dimgray',
            # bbox = dict(boxstyle = 'round,pad=0.3', facecolor = 'white', edgecolor = 'lightgray', alpha = 0.8)
        )

        # Only set Y-axis label on the leftmost plot
        if i == 0:
            ax.set_ylabel("Activation Frequency (%)", fontweight = "bold", fontsize = 14)
        else:
            ax.set_ylabel("")

        # Remove top and right spines for a cleaner look
        sns.despine(ax = ax, top = True, right = True)

        # Add subtle horizontal grid lines for readability
        ax.grid(axis = 'y', linestyle = '--', alpha = 0.3, color = '#CCCCCC')

        # Remove the per-subplot legend (unified legend is added below)
        ax.get_legend().remove()

    # Add super title for the entire figure
    plt.suptitle(
        "Top-10 Neuron Activation Frequencies: Target Word Present vs. Removed",
        fontsize = 18,
        fontweight = "bold",
        y = 1.03
    )

    # Add a single unified legend centered under the title, in row format
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc = 'upper center',
        bbox_to_anchor = (0.5, 0.9975),
        ncol = len(handles),
        frameon = False,
        prop = {'size': 12, 'weight': 'bold'}
    )

    # Show and save the plot
    plt.tight_layout()
    plt.savefig("lexical_comparison.png", dpi = 300, bbox_inches = "tight")
    plt.savefig("lexical_comparison.pdf", bbox_inches = "tight")
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
    df_removed = create_removed_df(df)

    # Measure activation in both conditions
    original_results = evaluate_neurons(df, tokenizer, model, top_neurons)
    removed_results = evaluate_neurons(df_removed, tokenizer, model, top_neurons)

    # Plot the comparison of neuron activation between original and modified datasets
    plot_comparison(original_results, removed_results)
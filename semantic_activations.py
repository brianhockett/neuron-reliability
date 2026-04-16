# Imports
from transformers import BertModel, BertTokenizer, logging
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch

# Suppress warnings from unused pre-training heads
logging.set_verbosity_error()

# Load STS-B dataset and filter by score range
def load_stsb(min_score = 0.8, max_score = 1.0, split = "validation"):
    
    dataset = load_dataset("sentence-transformers/stsb")
    
    if min_score is not None and max_score is not None:
        dataset = dataset[split].filter(lambda x: min_score <= x["score"] <= max_score)
    else:
        dataset = dataset[split]
    
    print(f"Loaded {len(dataset)} pairs with score {min_score} <= score <= {max_score}")
    return dataset


# Load BERT model and tokenizer
def load_model(model_name = "bert-base-uncased"):
    
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_hidden_states = True)
    model.eval()
    
    return tokenizer, model


# Compute all layer activations for a sentence (cache activations to avoid recomputation)
def compute_activations(sentence, tokenizer, model):
    
    inputs = tokenizer(sentence, return_tensors = "pt", truncation = True, max_length = 128)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # hidden_states[0] is embeddings; hidden_states[1:] are the 12 layers
    # Exclude [CLS] and [SEP] tokens
    return [layer.squeeze(0)[1:-1].numpy() for layer in outputs.hidden_states[1:]]


# Extract top-k neurons from precomputed activations for a given layer
def get_top_k_from_activations(layer_activations, layer_index, k = 50):
    
    neuron_scores = layer_activations[layer_index].max(axis = 0)  # max across tokens
    top_k_indices = set(np.argsort(neuron_scores)[-k:])
    
    return top_k_indices


# Compute intersection rate between two sets of top-k neuron indices
def intersection_rate(set_a, set_b, k = 50):
    return len(set_a & set_b) / k


# Run analysis for a single layer using cached activations
def run_layer_analysis(dataset_activations, layer_index, k = 50):
    
    similar_intersections = []
    random_intersections = []

    # Extract top-k neuron sets for all pairs at this layer
    top_k_sets = []
    for act1, act2 in dataset_activations:
        set1 = get_top_k_from_activations(act1, layer_index, k)
        set2 = get_top_k_from_activations(act2, layer_index, k)
        top_k_sets.append((set1, set2))

    # Calculate intersection rates for true similar pairs
    for set1, set2 in top_k_sets:
        similar_intersections.append(intersection_rate(set1, set2, k))

    # Generate random baseline by shuffling sentence2
    indices = np.random.permutation(len(top_k_sets))
    for i, j in enumerate(indices):
        random_intersections.append(intersection_rate(top_k_sets[i][0], top_k_sets[j][1], k))

    return {
        "similar": similar_intersections,
        "random": random_intersections
    }


# Run analysis across all layers using cached activations
def run_analysis_all_layers(min_score = 0.8, max_score = 1.0, k = 50):
    
    # Load dataset
    dataset = load_stsb(min_score, max_score)
    
    # Load BERT model and tokenizer
    tokenizer, model = load_model()
    num_layers = model.config.num_hidden_layers

    # Precompute activations for all sentences in the dataset
    dataset_activations = []
    for i, example in enumerate(dataset):
        act1 = compute_activations(example["sentence1"], tokenizer, model)
        act2 = compute_activations(example["sentence2"], tokenizer, model)
        dataset_activations.append((act1, act2))

        if (i + 1) % 50 == 0:
            print(f"Precomputed {i+1}/{len(dataset)} activations")

    # Run analysis for each layer
    results_by_layer = {}
    for layer_index in range(num_layers):
        print(f"Analyzing Layer {layer_index}")
        layer_results = run_layer_analysis(dataset_activations, layer_index, k)
        results_by_layer[layer_index] = layer_results

        # print(f"  Similar mean: {np.mean(layer_results['similar']):.4f}, "
        # f"Random mean: {np.mean(layer_results['random']):.4f}")

    return results_by_layer


# Plot intersection rate trends across layers
def plot_by_layer(results_by_layer, k):
    
    layers = sorted(results_by_layer.keys())
    
    sim_means = [np.mean(results_by_layer[l]['similar']) for l in layers]
    sim_stds  = [np.std(results_by_layer[l]['similar']) for l in layers]
    ran_means = [np.mean(results_by_layer[l]['random']) for l in layers]
    ran_stds  = [np.std(results_by_layer[l]['random']) for l in layers]

    plt.figure(figsize = (14, 6))

    # Plot similar pairs
    plt.plot(layers, sim_means, marker = 'o', color = 'steelblue', label = 'Similar Pairs', linewidth = 2.5)
    plt.fill_between(layers, 
                     np.array(sim_means) - np.array(sim_stds), 
                     np.array(sim_means) + np.array(sim_stds), 
                     color = 'steelblue', alpha = 0.2)

    # Plot random pairs
    plt.plot(layers, ran_means, marker = 's', color = 'salmon', label = 'Random Pairs', linewidth = 2.5)
    plt.fill_between(layers, 
                     np.array(ran_means) - np.array(ran_stds), 
                     np.array(ran_means) + np.array(ran_stds), 
                     color = 'salmon', alpha = 0.2)

    # Formatting
    plt.title(f'Top-{k} Neuron Intersection Rate by Layer', fontsize = 14, pad = 25, fontweight = 'bold')
    plt.xlabel('BERT Layer', fontsize = 12)
    plt.ylabel('Intersection Rate', fontsize = 12)
    plt.xticks(layers)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle = '--', alpha = 0.5)
    
    # Legend horizontally below the title, no frame
    plt.legend(fontsize = 11, loc = 'upper center', ncol = 2, frameon = False, bbox_to_anchor = (0.5, 1.08))
    
    sns.despine(left = False, bottom = False)  # remove top and right spines
    plt.tight_layout()
    plt.savefig("intersection_comparison.png", dpi = 150)
    plt.show()


# Plot boxplots of intersection distributions
def plot_boxplot(results_by_layer, k):
    
    # Flatten data into a DataFrame
    data = []
    for layer, res in results_by_layer.items():
        for val in res['similar']:
            data.append({'Layer': layer, 'Intersection Rate': val, 'Pair Type': 'Similar Pairs'})
        for val in res['random']:
            data.append({'Layer': layer, 'Intersection Rate': val, 'Pair Type': 'Random Pairs'})

    df = pd.DataFrame(data)

    plt.figure(figsize = (14, 6))
    sns.boxplot(data = df, x = 'Layer', y = 'Intersection Rate', hue = 'Pair Type',
                palette = {'Similar Pairs': 'steelblue', 'Random Pairs': 'salmon'},
                fliersize = 5, linewidth = 1.2, gap = 0.2)

    plt.title(f'Top-{k} Neuron Intersection Distributions by Layer', fontsize = 14, fontweight = 'bold')
    plt.ylabel('Intersection Rate', fontsize = 12)
    plt.xlabel('BERT Layer', fontsize = 12)
    plt.ylim(0, 1.05)

    # Legend horizontally below the title, no frame
    plt.legend(fontsize = 11, loc = 'upper center', ncol = 2, frameon = False, bbox_to_anchor = (0.5, 1.025))

    sns.despine(left = False, bottom = False)
    plt.tight_layout()
    plt.savefig("intersection_boxplot.png", dpi = 150)
    plt.show()


# Main execution
if __name__ == "__main__":
    # k = 5 for top-k neurons
    k = 5
    results = run_analysis_all_layers(min_score = 0.8, max_score = 1.0, k = k)
    plot_by_layer(results, k = k)
    plot_boxplot(results, k = k)
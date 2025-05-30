
import random
import numpy as np
import json
from collections import Counter
import math

# ------------------------------
# 1. Zipf Distribution
# ------------------------------

def zipf_distribution(size, alpha=1.1):
    ranks = np.arange(1, size + 1)
    weights = 1 / np.power(ranks, alpha)
    probabilities = weights / np.sum(weights)
    return probabilities

def weighted_choice(words, probabilities):
    return np.random.choice(words, p=probabilities)

# ------------------------------
# 2. Lexicon Generation
# ------------------------------

def generate_lexicon(vocab_sizes):
    lexicon = {}
    for pos, size in vocab_sizes.items():
        lexicon[pos] = [f"{pos}{i}" for i in range(1, size + 1)]
    return lexicon

# ------------------------------
# 3. Sentence Generators by Tier
# ------------------------------

def generate_basic_sentence(lexicon):
    noun_probs = zipf_distribution(len(lexicon["noun"]))
    verb_probs = zipf_distribution(len(lexicon["transitive_verb"]))

    subj = weighted_choice(lexicon["noun"], noun_probs)
    obj = weighted_choice(lexicon["noun"], noun_probs)
    verb = weighted_choice(lexicon["transitive_verb"], verb_probs)

    sentence = f"{subj.capitalize()} {verb}s the {obj}"
    semantics = f"∃e (Agent(e, {subj}) ∧ Patient(e, {obj}) ∧ Event(e, {verb}))"
    return sentence, semantics

def generate_transitive_event(lexicon):
    noun_probs = zipf_distribution(len(lexicon["noun"]))
    verb_probs = zipf_distribution(len(lexicon["transitive_verb"]))
    adv_probs = zipf_distribution(len(lexicon["adverb"]))

    subj = weighted_choice(lexicon["noun"], noun_probs)
    obj = weighted_choice(lexicon["noun"], noun_probs)
    verb = weighted_choice(lexicon["transitive_verb"], verb_probs)
    adverb = weighted_choice(lexicon["adverb"], adv_probs)

    sentence = f"The {subj} {verb}s the {obj} {adverb}"
    semantics = f"∃e (Agent(e, {subj}) ∧ Patient(e, {obj}) ∧ Manner(e, {adverb}) ∧ Event(e, {verb}))"
    return sentence, semantics

def generate_complex_event(lexicon):
    base_sentence, semantics = generate_transitive_event(lexicon)
    verb_probs = zipf_distribution(len(lexicon["transitive_verb"]))
    noun_probs = zipf_distribution(len(lexicon["noun"]))

    embedded_verb = weighted_choice(lexicon["transitive_verb"], verb_probs)
    embedded_obj = weighted_choice(lexicon["noun"], noun_probs)
    clause = f"who {embedded_verb}ed the {embedded_obj}"

    tokens = base_sentence.split()
    tokens.insert(1, clause)
    sentence = " ".join(tokens)
    semantics += f" ∧ Clause(e, {clause.replace(' ', '_')})"
    return sentence, semantics

def generate_sentence_with_tier(tier, lexicon):
    if tier == 1:
        return generate_basic_sentence(lexicon)
    elif tier == 2:
        return generate_transitive_event(lexicon)
    elif tier == 3:
        return generate_complex_event(lexicon)
    else:
        raise ValueError("Tier must be 1, 2, or 3.")

# ------------------------------
# 4. Corpus Generation
# ------------------------------

def generate_corpus_with_tiers(lexicon, counts_per_tier):
    corpus = []
    for tier, count in counts_per_tier.items():
        for _ in range(count):
            sentence, semantics = generate_sentence_with_tier(tier, lexicon)
            corpus.append({
                "tier": tier,
                "sentence": sentence,
                "semantics": semantics
            })
    return corpus

# ------------------------------
# 5. Evaluation Metrics
# ------------------------------

def token_entropy(sentences):
    token_counts = Counter()
    for sent in sentences:
        tokens = sent.split()
        token_counts.update(tokens)
    total = sum(token_counts.values())
    probs = [count / total for count in token_counts.values()]
    return -sum(p * math.log2(p) for p in probs)

def type_token_ratio(sentences):
    tokens = [token for sent in sentences for token in sent.split()]
    types = set(tokens)
    return len(types) / len(tokens)

def compute_metrics(corpus):
    sentences = [entry["sentence"] for entry in corpus]
    return {
        "token_entropy": token_entropy(sentences),
        "type_token_ratio": type_token_ratio(sentences),
        "total_sentences": len(sentences),
        "tiers": Counter(entry["tier"] for entry in corpus)
    }

# ------------------------------
# 6. Main Execution
# ------------------------------

if __name__ == "__main__":
    vocab_sizes = {
        "noun": 100,
        "transitive_verb": 30,
        "adverb": 15
    }

    lexicon = generate_lexicon(vocab_sizes)

    tier_counts = {1: 100, 2: 100, 3: 100}
    corpus = generate_corpus_with_tiers(lexicon, tier_counts)

    with open("corpus_tiered.json", "w") as f:
        json.dump(corpus, f, indent=2)

    metrics = compute_metrics(corpus)
    print("Corpus Evaluation Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

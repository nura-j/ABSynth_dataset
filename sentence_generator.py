import random
import math
from collections import Counter

class SentenceGenerator:
    """
    Generates sentences with controlled linguistic properties
    for next token prediction tasks.
    """
    
    def __init__(self, lexicon_generator, template_manager):
        """
        Initialize with lexicon and template components.
        
        Args:
            lexicon_generator: Instance of LexiconGenerator
            template_manager: Instance of TemplateManager
        """
        self.lexicon = lexicon_generator
        self.templates = template_manager
        self.bigram_counts = Counter()
        self.sentence_lengths = []

    def _expand_template_tag(self, tag, context=None):
        """
        Expand a template tag into an actual word.

        Args:
            tag: Template tag (e.g., "NOUN", "ADJ", "VERB")
            context: Optional context information

        Returns:
            The expanded word
        """
        # Map template tags to word categories
        tag_mapping = {
            "NOUN": "noun",
            "VERB": random.choice(["transitive_verb", "intransitive_verb", "motion_verb", "communication_verb"]),
            "TVERB": "transitive_verb",
            "IVERB": "intransitive_verb",
            "MVERB": "motion_verb",
            "CVERB": "communication_verb",
            "ADJ": "adjective",
            "ADV": "adverb",
            "PREP": "preposition",  # For regular prepositions
            "LOCATION": "location",  # Explicit mapping for location category
            "DET": "determiner",
            "CONJ": "conjunction",
            "REL": "rel",
            "COMP": "comp",
            "TEMP": "temporal",
            "RESULT": "result"  # Added result category
        }

        category = tag_mapping.get(tag, tag)

        word = self.lexicon.sample_word(category, context)

        # Handle verb conjugation (slightly more varied)
        if tag == "VERB" or tag == "TVERB" or tag == "IVERB" or tag == "MVERB" or tag == "CVERB":
            if context and context.get("tense") == "past":
                # Add some irregularity to make more interesting patterns
                if random.random() < 0.1:  # 10% chance of irregular verb
                    return f"{word}{random.choice(['t', 'ed', 'ew', 'ook'])}"
                return f"{word}ed"
            else:
                # Present tense - occasionally use different endings
                if random.random() < 0.1:  # 10% chance of variation
                    return word  # Base form (for modal constructions)
                return f"{word}s"

        return word


    def generate_sentence(self, complexity=None, include_metadata=True):
        """
        Generate a single sentence with controlled linguistic properties.

        Args:
            complexity: Optional specific complexity level ("simple", "medium", "complex")

        Returns:
            Dictionary with sentence, semantic representation, and metadata
        """
        # Select template based on complexity distribution
        if complexity:
            weights = {c: 1.0 if c == complexity else 0.0 for c in ["simple", "medium", "complex"]}
            template, complexity_level = self.templates.select_template(weights)
        else:
            template, complexity_level = self.templates.select_template()

        words = []
        semantics_dict = {}
        context = {"head": None, "tense": "present"}

        # Generate sentence by expanding template tags
        for i, tag in enumerate(template):
            # Update context based on sentence structure
            if i > 0 and tag == "NOUN" and template[i-1] == "DET":
                # Reset head when we start a new noun phrase
                context["head"] = None

            # Expand tag to actual word
            word = self._expand_template_tag(tag, context)
            words.append(word)

            # Update context for next words
            if tag == "NOUN":
                context["head"] = word  # Noun becomes head for adjectives
                if "Agent" not in semantics_dict:
                    semantics_dict["Agent"] = word
                elif "Patient" not in semantics_dict:
                    semantics_dict["Patient"] = word
            elif tag == "VERB" or tag == "TVERB":
                if "Predicate" not in semantics_dict:
                    semantics_dict["Predicate"] = word
                context["head"] = word  # Verb becomes head for objects

            # Update bigram statistics
            if i > 0:
                bigram = (words[i-1], word)
                self.bigram_counts[bigram] += 1

        # Format the sentence with proper spacing and capitalization
        formatted_words = []
        for i, word in enumerate(words):
            if i > 0 and word == ",":
                # No space before comma
                formatted_words[-1] = formatted_words[-1] + word
            elif i > 0 and words[i-1] == ",":
                # Space after comma
                formatted_words.append(word)
            # elif i == 0:
            #     # Capitalize first word
            #     formatted_words.append(word.capitalize())
            else:
                formatted_words.append(word)

        sentence = " ".join(formatted_words)

        # Record sentence length for statistics
        self.sentence_lengths.append(len(words))

        # Create simplified semantics representation
        semantics = f"∃e ({' ∧ '.join([f'{k}(e, {v})' for k, v in semantics_dict.items()])})"

        # Calculate entropy profile for this sentence
        entropy_profile = self._calculate_entropy_profile(words)

        # Create metadata with information useful for next token prediction evaluation
        metadata = {
            "complexity": complexity_level,
            "template": str(template),
            "length": len(words),
            "entropy_profile": entropy_profile,
            "avg_entropy": sum(entropy_profile) / len(entropy_profile) if entropy_profile else 0
        }
        if include_metadata:

            result = {
                "sentence": sentence,
                "semantics": semantics,
                "metadata": metadata
            }
        else:
            result = {
                "sentence": sentence,
                "semantics": semantics
            }

        return result

    # In sentence_generator.py, modify the _calculate_entropy_profile method slightly:

    def _calculate_entropy_profile(self, words):
        """Calculate the entropy profile with more medium-predictability values."""
        entropy_profile = []

        # For each position, calculate entropy of potential next words
        for i in range(len(words) - 1):
            current_word = words[i]

            # Find all instances of current_word and count following words
            following_counts = Counter()
            total_count = 0

            for bg, count in self.bigram_counts.items():
                if bg[0] == current_word:
                    following_counts[bg[1]] += count
                    total_count += count

            # Calculate entropy if we have seen this word before
            if total_count > 0:
                probs = [count / total_count for count in following_counts.values()]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)

                # Adjust entropy toward the medium range if it's too extreme
                if entropy < 1.0:  # Too predictable
                    entropy = 1.0 + (entropy * 0.5)  # Shift toward medium (1.0-2.5)
                elif entropy > 3.0:  # Too unpredictable
                    entropy = 3.0 - ((entropy - 3.0) * 0.5)  # Shift toward medium

                entropy_profile.append(entropy)
            else:
                # If we haven't seen this word, assign medium entropy
                entropy_profile.append(2.0)  # Medium predictability instead of 2.5

        # For the last position, assign medium-high entropy
        entropy_profile.append(2.8)  # Reduced from 3.5

        return entropy_profile
    
    def get_predictability_distribution(self, entropy_profiles):
        """
        Calculate the distribution of predictability levels in the corpus.
        
        Args:
            entropy_profiles: List of entropy profiles from generated sentences
        
        Returns:
            Dictionary with predictability distribution
        """
        entropy_values = []
        for profile in entropy_profiles:
            entropy_values.extend(profile)
        
        if not entropy_values:
            return {
                "high_predictability": 0, 
                "medium_predictability": 0, 
                "low_predictability": 0
            }
        
        # Categorize by predictability
        high_pred = sum(1 for e in entropy_values if e < 1.5) / len(entropy_values)
        medium_pred = sum(1 for e in entropy_values if 1.5 <= e < 3.0) / len(entropy_values)
        low_pred = sum(1 for e in entropy_values if e >= 3.0) / len(entropy_values)
        
        return {
            "high_predictability": high_pred,
            "medium_predictability": medium_pred,
            "low_predictability": low_pred
        }
    
    def get_bigram_statistics(self):
        """
        Get statistics about bigram distribution in generated sentences.
        
        Returns:
            Dictionary with bigram statistics
        """
        total_bigrams = sum(self.bigram_counts.values())
        unique_bigrams = len(self.bigram_counts)
        
        bigram_diversity = unique_bigrams / total_bigrams if total_bigrams > 0 else 0
        
        # Top bigrams with counts
        top_bigrams = self.bigram_counts.most_common(20)
        
        # Calculate mutual information for common bigrams
        mutual_info = {}
        word_counts = Counter()
        
        # Count individual words
        for (w1, w2), count in self.bigram_counts.items():
            word_counts[w1] += count
            word_counts[w2] += count
        
        # Calculate mutual information
        for (w1, w2), joint_prob in self.bigram_counts.most_common(50):
            joint_prob = joint_prob / total_bigrams
            w1_prob = word_counts[w1] / (total_bigrams + len(self.sentence_lengths))
            w2_prob = word_counts[w2] / (total_bigrams + len(self.sentence_lengths))
            
            # Mutual information formula: log2(P(w1,w2) / (P(w1) * P(w2)))
            if joint_prob > 0 and w1_prob > 0 and w2_prob > 0:
                mi = math.log2(joint_prob / (w1_prob * w2_prob))
                mutual_info[(w1, w2)] = mi
        
        return {
            "total_bigrams": total_bigrams,
            "unique_bigrams": unique_bigrams,
            "bigram_diversity": bigram_diversity,
            "top_bigrams": top_bigrams,
            "mutual_information": mutual_info
        }
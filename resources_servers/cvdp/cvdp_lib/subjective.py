# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# This file is replicated from the CVDP benchmark repository.

import re

import nltk.translate.bleu_score as bleu


def calculate_BLEU(generated_summary, reference_summary, n):
    # Tokenize the generated summary and reference summary
    generated_tokens = generated_summary.split()
    reference_tokens = reference_summary.split()

    # Calculate the BLEU score
    weights = [1.0 / n] * n  # Weights for n-gram precision calculation
    bleu_score = bleu.sentence_bleu([reference_tokens], generated_tokens, weights=weights)

    return bleu_score


def calculate_ROUGE(generated_summary, reference_summary, n):
    # Tokenize the generated summary and reference summary into n-grams
    generated_ngrams = generate_ngrams(generated_summary, n)
    reference_ngrams = generate_ngrams(reference_summary, n)

    # Calculate the recall score
    matching_ngrams = len(set(generated_ngrams) & set(reference_ngrams))
    recall_score = matching_ngrams / len(reference_ngrams)

    return recall_score


def generate_ngrams(text, n):
    # Preprocess text by removing punctuation and converting to lowercase
    text = re.sub(r"[^\w\s]", "", text.lower())

    # Generate n-grams from the preprocessed text
    words = text.split()
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]

    return ngrams


if __name__ == "__main__":
    # Example usage
    generated_summary = "The dog slept on the couch."
    reference_summary = "The cat sat on the mat."
    n = 2  # bigram

    rouge_score = calculate_ROUGE(generated_summary, reference_summary, n)
    print(f"ROUGE-{n} score: {rouge_score}")

    bleu_score = calculate_BLEU(generated_summary, reference_summary, n)
    print(f"BLEU-{n} score: {bleu_score}")

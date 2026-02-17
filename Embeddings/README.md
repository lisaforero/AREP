# LLM Text Preprocessing Foundations (Embeddings)

This notebook explores the fundamental stages of an LLM data pipeline, transforming raw text into structured vector representations suitable for GPT-like architectures.

## Project description

This project implements a complete data preprocessing workflow for Large Language Models. It covers text tokenization using Byte Pair Encoding (BPE), the creation of efficient Data Loaders using a sliding window approach, and the implementation of Token and Positional Embeddings. The goal is to understand how raw data becomes a mathematical representation that an agentic system can process.

This implementation follows the guide found in Chapter 2 of "Build a Large Language Model (From Scratch)" by Sebastian Raschka.

__Notebook:__ https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/ch02.ipynb

__Text:__ https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt

## Overview

In this notebook, through explanatory cells, I analyzed how embeddings encode semantic meaning through their arrangement in a vector space and their relationship with weight matrices in neural networks. Finally, I conducted an experiment adjusting the context window (max_length) and the stride, demonstrating that overlap is fundamental to generate more training samples and ensure the logical continuity of the language, preventing the model from losing the thread between data blocks.

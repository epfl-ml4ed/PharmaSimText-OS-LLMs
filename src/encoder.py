import json
import os
import numpy as np
import fasttext
import fasttext.util
import shutil
from sentence_transformers import SentenceTransformer


class Encoder:
    """
    Encoder class for handling text embeddings with support for FastText and BERT-based models.

    Parameters:
    - encoder_type (str): The type of encoder to use ("fasttext", "bert", or "medbert").
    - emb_dim (int): Desired embedding dimension.
    - memory_size (int): Maximum number of encodings to store in memory.
    - save_memory (float): Percentage of memory to save incrementally.
    - memory_path (str): Path to JSON file for saving memory cache.
    """

    def __init__(self, encoder_type="fasttext", emb_dim=300, memory_size=1000, save_memory=0.1,
                 memory_path='../encoder_memory/memory.json'):
        self.memory_path = memory_path
        self.memory_size = memory_size
        self.save_memory = save_memory
        self.encoder_type = encoder_type
        self.memory = {}

        # Ensure memory path directory exists
        os.makedirs(os.path.dirname(memory_path), exist_ok=True)

        # Initialize encoder model based on encoder_type
        if encoder_type == "fasttext":
            self._initialize_fasttext(emb_dim)
        elif encoder_type in ["bert", "medbert"]:
            self._initialize_bert(encoder_type)
        else:
            raise NotImplementedError(f"Encoder type '{encoder_type}' is not supported.")

        # Load memory from file if it exists
        if os.path.exists(memory_path):
            with open(memory_path, 'r') as f:
                self.memory = json.load(f)

    def _initialize_fasttext(self, emb_dim):
        """Initializes the FastText model, downloading and resizing if necessary."""
        fasttext_model_path = "./lms/cc.en.300.bin"

        # Download and move FastText model if not already present
        if not os.path.exists(fasttext_model_path):
            fasttext.util.download_model('en')
            shutil.move("./cc.en.300.bin", fasttext_model_path)

        self.model = fasttext.load_model(fasttext_model_path)

        # Reduce FastText model dimension if specified
        if emb_dim < 300:
            print("Reducing FastText model dimension...")
            fasttext.util.reduce_model(self.model, emb_dim)

    def _initialize_bert(self, encoder_type):
        """Initializes the SentenceTransformer model based on BERT/MedBERT model type."""
        model_name = 'all-MiniLM-L6-v2' if encoder_type == "bert" else 'pritamdeka/S-PubMedBert-MS-MARCO'
        self.model = SentenceTransformer(model_name)

    def encode(self, sentences):
        """
        Encodes a list of sentences and caches them in memory.

        Parameters:
        - sentences (list of str): Sentences to encode.

        Returns:
        list of np.array: Encoded sentence vectors.
        """
        encoded = []
        for sentence in sentences:
            # Encode and cache if sentence not already in memory
            if sentence not in self.memory:
                vector = self._encode_sentence(sentence)
                if len(self.memory) + 1 > self.memory_size:
                    self.memory.popitem()  # Remove oldest item if memory is full
                self.memory[sentence] = vector.tolist()

                # Save memory periodically
                if len(self.memory) % round(self.memory_size * self.save_memory) == 0:
                    self._save_memory()

            encoded.append(np.array(self.memory[sentence]))
        return encoded

    def _encode_sentence(self, sentence):
        """Encodes a single sentence based on the encoder type."""
        if self.encoder_type == "fasttext":
            return self.model.get_sentence_vector(sentence).reshape(1, -1)
        return self.model.encode([sentence]).reshape(1, -1)

    def _save_memory(self):
        """Saves the current memory cache to a JSON file."""
        with open(self.memory_path, 'w') as f:
            json.dump(self.memory, f)


def main():
    """
    Main function to demonstrate the Encoder class with sample sentences.
    """
    # Initialize encoder with specified parameters
    encoder = Encoder(
        encoder_type="medbert",
        emb_dim=300,
        memory_size=10,
        save_memory=0.1,
        memory_path='../encoder_memory/medbert.json'
    )

    # List of sample sentences to encode
    sample_sentences = [
        "The mysterious old book sat on the dusty shelf, its pages filled with forgotten tales.",
        "With a sudden gust of wind, the leaves danced in a chaotic frenzy, painting the autumn sky with shades of gold and crimson.",
        "As the first rays of sunlight peeked over the horizon, the sleepy town slowly awakened to a new day.",
    ]

    # Encode sentences and print the resulting embeddings
    embeddings = encoder.encode(sample_sentences)
    print(embeddings)
    print(f"Shape of first embedding: {embeddings[0].shape}")

if __name__ == '__main__':
    main()


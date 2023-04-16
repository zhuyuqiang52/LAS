
class SimpleTokenizer():
    '''
    Partially written by ChatGPT
    '''
    def __init__(self, vocab, eos_idx):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.eos_idx = eos_idx
    
    # Encode a text string as a list of integers
    def encode(self, text, eos=True):
        encoded = []
        for char in text.upper():
            if char in self.vocab:
                encoded.append(self.vocab.index(char))
        if eos:
            encoded += [self.eos_idx]

        return encoded

    # Decode a list of integers as a text string
    def decode(self, encoded):
        decoded = ""
        for idx in encoded:
            if idx == self.eos_idx:
                break
            if idx < self.vocab_size:
                decoded += self.vocab[idx]
           
        return decoded
 
"""A simple Bigram language example in JAX."""

import flax.linen as nn


class BigramLanguageModel(nn.Module):
  """A simple Bigram language model.

  Attributes:
    token_embedding_table:
    vocab_size:
  """
  vocab_size: int

  def setup(self):
    # Each token directly reads off the logits for the next token from a lookup
    # table
    self.token_embedding_table = nn.Embed(self.vocab_size, self.vocab_size)

  @nn.compact
  def __call__(self, idx):
    # idx are (B, T) tensor of integers, C is vocab_size here.
    logits = self.token_embedding_table(idx)  # (B,T,C)

    return logits
  

"""Characterization tests for arnold.torch_model.sensorimotor_vocabulary."""

import torch
import pytest
import numpy as np

import helpers

from arnold.torch_model.sensorimotor_vocabulary import SensorimotorVocabulary


########################################
#              Vocabulary              #
########################################

def _build():
    torch.manual_seed(0)
    return SensorimotorVocabulary(embed_dim=helpers.EMBED_DIM)


def test_vocab_size_and_no_duplicates(goldens):
    vocab = _build()
    assert vocab.vocab_size == goldens["vocab_size"]
    assert vocab.n_tokens == len(vocab.all_tokens)
    # token_to_idx is a bijection over all_tokens (no collisions)
    assert len(set(vocab.all_tokens)) == len(vocab.all_tokens)
    assert len(vocab.token_to_idx) == vocab.n_tokens


def test_get_embedding_is_sum_of_token_embeddings():
    vocab = _build()
    sig = ("DELT1", "r", "muscle", "activation")
    emb = vocab.get_embedding(sig)
    manual = sum(
        vocab.embeddings.weight[vocab.token_to_idx[t]] for t in sig
    )
    assert torch.allclose(emb, manual, rtol=1e-6, atol=1e-6)
    assert emb.shape == (helpers.EMBED_DIM,)


def test_get_embedding_matches_golden(goldens_npz):
    vocab = _build()
    emb_a = vocab.get_embedding(("DELT1", "r", "muscle", "activation"))
    emb_b = vocab.get_embedding(("femur", "l", "position", "x"))
    np.testing.assert_allclose(
        emb_a.detach().numpy(), goldens_npz["vocab_emb_a"], rtol=1e-6, atol=1e-6
    )
    np.testing.assert_allclose(
        emb_b.detach().numpy(), goldens_npz["vocab_emb_b"], rtol=1e-6, atol=1e-6
    )


def test_get_embedding_batch_equivalence_and_golden(goldens_npz):
    vocab = _build()
    sig_a = ("DELT1", "r", "muscle", "activation")
    sig_b = ("femur", "l", "position", "x")
    batch = vocab.get_embedding_batch([sig_a, sig_b, sig_a])
    assert batch.shape == (3, helpers.EMBED_DIM)
    # batch rows equal per-signature get_embedding
    stacked = torch.stack([
        vocab.get_embedding(sig_a),
        vocab.get_embedding(sig_b),
        vocab.get_embedding(sig_a),
    ])
    assert torch.allclose(batch, stacked, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        batch.detach().numpy(), goldens_npz["vocab_batch"], rtol=1e-6, atol=1e-6
    )


def test_unknown_token_raises():
    vocab = _build()
    with pytest.raises(KeyError):
        vocab.get_embedding(("NOT_A_TOKEN",))
    with pytest.raises(KeyError):
        vocab.get_embedding_batch([("NOT_A_TOKEN",)])

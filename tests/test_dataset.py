import torch
from integrate_zero.data.dataset import IntegrationDataset
from integrate_zero.data.vocabulary import Vocabulary


def test_dataset_length():
    ds = IntegrationDataset(num_samples=100, max_depth=3)
    assert len(ds) == 100


def test_dataset_item_shape():
    ds = IntegrationDataset(num_samples=10, max_depth=3)
    item = ds[0]
    assert "input_ids" in item
    assert "target_ids" in item
    assert "value_label" in item
    assert isinstance(item["input_ids"], torch.Tensor)
    assert item["input_ids"].dtype == torch.long


def test_dataset_input_starts_with_bos():
    vocab = Vocabulary()
    ds = IntegrationDataset(num_samples=10, max_depth=3)
    item = ds[0]
    assert item["input_ids"][0].item() == vocab.token_to_id("BOS")


def test_dataset_has_sep_token():
    vocab = Vocabulary()
    ds = IntegrationDataset(num_samples=10, max_depth=3)
    item = ds[0]
    sep_id = vocab.token_to_id("SEP")
    assert sep_id in item["input_ids"].tolist()


def test_collate_pads_to_same_length():
    ds = IntegrationDataset(num_samples=20, max_depth=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)
    batch = next(iter(loader))
    assert batch["input_ids"].shape[0] == 4
    assert batch["input_ids"].shape[1] > 0


def test_dataset_ends_with_eos():
    """Verify the input sequence ends with EOS token."""
    vocab = Vocabulary()
    ds = IntegrationDataset(num_samples=10, max_depth=3)
    item = ds[0]
    eos_id = vocab.token_to_id("EOS")
    assert item["input_ids"][-1].item() == eos_id


def test_target_ids_masked_before_sep():
    """Verify target_ids has -100 for all positions up to and including SEP."""
    vocab = Vocabulary()
    ds = IntegrationDataset(num_samples=10, max_depth=3)
    item = ds[0]
    sep_id = vocab.token_to_id("SEP")
    input_ids = item["input_ids"].tolist()
    target_ids = item["target_ids"].tolist()

    # Find SEP position
    sep_pos = input_ids.index(sep_id)

    # All positions up to and including SEP should be -100
    for i in range(sep_pos + 1):
        assert target_ids[i] == -100, f"Position {i} should be -100 but got {target_ids[i]}"

    # Positions after SEP should be actual token IDs (not -100)
    for i in range(sep_pos + 1, len(target_ids)):
        assert target_ids[i] != -100, f"Position {i} should not be -100"


def test_value_label_is_one():
    """All generated pairs are solvable, so value_label should be 1.0."""
    ds = IntegrationDataset(num_samples=10, max_depth=3)
    for i in range(len(ds)):
        item = ds[i]
        assert item["value_label"] == 1.0


def test_collate_target_ids_padded_with_negative_100():
    """Padded positions in target_ids should be -100, not 0."""
    ds = IntegrationDataset(num_samples=20, max_depth=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)
    batch = next(iter(loader))
    # Check that padding positions in target_ids are -100
    pad_mask = batch["input_ids"] == 0  # PAD id is 0
    if pad_mask.any():
        assert (batch["target_ids"][pad_mask] == -100).all()


def test_collate_value_label_shape():
    """Collated value_label should be a 1D tensor of length batch_size."""
    ds = IntegrationDataset(num_samples=20, max_depth=3)
    loader = torch.utils.data.DataLoader(ds, batch_size=4, collate_fn=ds.collate_fn)
    batch = next(iter(loader))
    assert batch["value_label"].shape == (4,)
    assert batch["value_label"].dtype == torch.float

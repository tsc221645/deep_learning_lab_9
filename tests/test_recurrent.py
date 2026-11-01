import torch

from src.models.recurrent import LSTMMultiOutput, LSTMRecursive, LSTMSeq2Seq


def test_recurrent_strategies_return_one_day_per_client():
    history = torch.randn(2, 48, 8)
    future = torch.randn(2, 24, 7)
    for cls in (LSTMRecursive, LSTMMultiOutput, LSTMSeq2Seq):
        model = cls(48, 24, 7, hidden_size=8, num_layers=1, dropout=0)
        assert model(history, future).shape == (2, 24)


def test_seq2seq_teacher_forcing_only_affects_training_path():
    torch.manual_seed(3)
    model = LSTMSeq2Seq(48, 24, 7, hidden_size=8, num_layers=1,
                        dropout=0, teacher_forcing=1)
    history = torch.randn(2, 48, 8)
    future = torch.randn(2, 24, 7)
    target_a = torch.zeros(2, 24)
    target_b = torch.ones(2, 24)
    assert not torch.equal(
        model.forward_train(history, future, target_a),
        model.forward_train(history, future, target_b),
    )
    assert torch.equal(model(history, future), model(history, future))

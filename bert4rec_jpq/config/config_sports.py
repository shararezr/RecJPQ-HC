from config import BERT4RecExperimentConfig


config = BERT4RecExperimentConfig(
    dataset_name='sports',
    sequence_length=50,
    embedding_size=64,

    # mapped params
    num_attention_heads=2,      # from num_heads
    num_hidden_layers=2,        # from num_blocks

    attention_probs_dropout_prob=0.2,  # from dropout_rate
    hidden_dropout_prob=0.6,           # from dropout_rate

    max_steps_per_epoch=128,   # from max_batches_per_epoch

    pq_m=8,
    gbce_t=0,
    codes_strategy="svd_opq",
)
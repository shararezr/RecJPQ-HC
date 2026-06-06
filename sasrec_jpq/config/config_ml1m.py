from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='ml1m',
    sequence_length=200,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=128,
    num_blocks=2,
    dropout_rate=0.2,
    negs_per_pos=1,
    gbce_t = 0,
    reuse_item_embeddings=True,
    codes_strategy = "bpr",
    pq_m=32,
    loss='gbce',
)

'''
config = GSASRecExperimentConfig(
    dataset_name='ml1m',
    sequence_length=50,
    embedding_dim=128,
    num_heads=1,
    max_batches_per_epoch=128,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=1,
    gbce_t = 0,
    reuse_item_embeddings=False,
)
'''
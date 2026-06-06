from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='beauty',
    sequence_length=50,
    embedding_dim=256,
    num_heads=1,
    max_batches_per_epoch=128,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=1,
    gbce_t =0,
    reuse_item_embeddings=True,
    codes_strategy = "svd",
    pq_m=32, 
    loss='gbce',
    use_jpq=True,
)

'''
config = GSASRecExperimentConfig(
    dataset_name='beauty',
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
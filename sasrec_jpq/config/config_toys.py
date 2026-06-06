from config import GSASRecExperimentConfig

config = GSASRecExperimentConfig(
    dataset_name='toys',
    sequence_length=50,
    embedding_dim=256,
    num_heads=1,
    max_batches_per_epoch=128,
    num_blocks=2,
    dropout_rate=0.5,
    negs_per_pos=1,
    gbce_t = 0,
    reuse_item_embeddings=True,
    codes_strategy = "bpr",
    pq_m=32,
    use_jpq=True,
    loss='gbce',
)

'''
config = GSASRecExperimentConfig(
    dataset_name='toys',
    sequence_length=50,
    embedding_dim=256,       
    num_heads=1,
    num_blocks=2,
    dropout_rate=0.5,        
    negs_per_pos=100,         
    gbce_t=0,              
    reuse_item_embeddings=True,
    codes_strategy="svd_opq",
    pq_m=16,                 
    max_batches_per_epoch=128
)



config = GSASRecExperimentConfig(
    dataset_name='toys',
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
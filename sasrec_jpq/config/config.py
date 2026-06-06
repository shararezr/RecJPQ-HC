from ir_measures import nDCG, R

class GSASRecExperimentConfig(object):
    def __init__(self, dataset_name, sequence_length=50, embedding_dim=256, train_batch_size=128,
                             num_heads=1, num_blocks=2,
                             dropout_rate=0.5,
                             negs_per_pos=1,
                             max_epochs=10000,
                             max_batches_per_epoch=100,
                             metrics=[nDCG@10, R@10, R@1],
                             val_metric = nDCG@10,
                             early_stopping_patience=200,
                             gbce_t =0,
                             filter_rated=True,
                             eval_batch_size=512,
                             recommendation_limit=10,
                             reuse_item_embeddings=True,
                             pq_m = 16,
                             strategy_weights= None,
                             strategy_partitions = None,
                             codes_strategy ='svd',
                             use_jpq=True,
                             loss='ce',
                             learning_rate=1e-3,
                             seed=42,
                             ):
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_blocks = num_blocks
        self.dropout_rate = dropout_rate
        self.dataset_name = dataset_name
        self.train_batch_size = train_batch_size
        self.negs_per_pos = negs_per_pos
        self.max_epochs = max_epochs
        self.max_batches_per_epoch = max_batches_per_epoch
        self.val_metric = val_metric
        self.metrics = metrics
        self.early_stopping_patience = early_stopping_patience
        self.gbce_t = gbce_t
        self.filter_rated = filter_rated
        self.recommendation_limit = recommendation_limit
        self.eval_batch_size = eval_batch_size
        self.reuse_item_embeddings = reuse_item_embeddings
        self.pq_m = pq_m
        self.strategy_weights= strategy_weights
        self.strategy_partitions =strategy_partitions
        self.codes_strategy = codes_strategy
        self.use_jpq = use_jpq
        self.loss = loss
        self.learning_rate = learning_rate
        self.seed = seed

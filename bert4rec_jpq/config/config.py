from ir_measures import nDCG, R


class BERT4RecExperimentConfig(object):
    def __init__(self,
                 dataset_name,
                 sequence_length=50,
                 embedding_size=64,
                 num_attention_heads=2,
                 num_hidden_layers=2,
                 hidden_act="gelu",
                 attention_probs_dropout_prob=0.2,
                 hidden_dropout_prob=0.6,
                 initializer_range=0.02,
                 type_vocab_size=2,
                 masking_prob=0.2,
                 train_batch_size=128,
                 eval_batch_size=128,
                 max_steps_per_epoch=128,
                 max_epochs=10000,
                 early_stopping_patience=200,
                 lr=0.001,
                 metrics=[nDCG@10, R@10, R@1],
                 val_metric=nDCG@10,
                 recommendation_limit=10,
                 filter_rated=True,
                 pq_m=16,
                 codes_strategy='svd',
                 strategy_weights=None,
                 strategy_partitions=None,
                 seed=42,
                 loss='ce',
                 negs_per_pos=1,
                 gbce_t=0
                 ):
        self.dataset_name = dataset_name
        self.sequence_length = sequence_length
        self.embedding_size = embedding_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.hidden_act = hidden_act
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.initializer_range = initializer_range
        self.type_vocab_size = type_vocab_size
        self.masking_prob = masking_prob
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.max_steps_per_epoch = max_steps_per_epoch
        self.max_epochs = max_epochs
        self.early_stopping_patience = early_stopping_patience
        self.lr = lr
        self.metrics = metrics
        self.val_metric = val_metric
        self.recommendation_limit = recommendation_limit
        self.filter_rated = filter_rated
        self.pq_m = pq_m
        self.codes_strategy = codes_strategy
        self.strategy_weights = strategy_weights
        self.strategy_partitions = strategy_partitions
        self.seed = seed
        self.loss = loss
        self.negs_per_pos=negs_per_pos
        self.gbce_t = gbce_t

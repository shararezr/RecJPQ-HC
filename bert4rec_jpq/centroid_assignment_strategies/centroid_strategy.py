class CentroidAssignmentStragety(object):
    def __init__(self, 
                 item_code_bytes, 
                 num_items,
                 sub_embedding_size,
                 codes_strategy="svd",
                 strategy_weights=None,
                 strategy_partitions=None):
        
        self.item_code_bytes = item_code_bytes
        self.num_items = num_items
        self.sub_embedding_size = sub_embedding_size
        self.codes_strategy = codes_strategy
        self.strategy_weights = strategy_weights
        self.strategy_partitions = strategy_partitions
        
    
    def assign(self, train_users):
        raise NotImplementedError()

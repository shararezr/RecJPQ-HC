from tqdm import tqdm
from aprec.api.item import Item

from aprec.api.items_ranking_request import ItemsRankingRequest
from aprec.api.user import User
from aprec.api.action import Action


class Recommender():
    def __init__(self):
        self.items_ranking_requests = []
        self.val_users = set()

    def name(self):
        raise NotImplementedError

    def add_action(self, action: Action):
        raise (NotImplementedError)

    def rebuild_model(self):
        raise (NotImplementedError)

    def recommend(self, user_id, limit: int, features=None):
        raise (NotImplementedError)

    # recommendation request = tuple(user_id, features)
    def recommend_batch(self, recommendation_requests, limit):
        results = []
        for user_id, features in tqdm(recommendation_requests, ascii=True):
            results.append(self.recommend(user_id, limit, features))
        return results

    # many recommenders don't require users, so leave it doing nothing by default
    def add_user(self, user: User):
        pass

    # many recommenders don't require items, so leave it doing nothing by default
    def add_item(self, item: Item):
        pass

    def recommend_by_items(self, items_list, limit: int):
        raise (NotImplementedError)

    def get_similar_items(self, item_id, limit: int):
        raise (NotImplementedError)

    def to_str(self):
        raise (NotImplementedError)

    def from_str(self):
        raise (NotImplementedError)

    #the directory where the recommender can save stuff, like logs
    def set_out_dir(self, out_dir):
        self.out_dir = out_dir

    def get_metadata(self):
        return {}

    def set_val_users(self, val_users):
        self.val_users = val_users

    def add_test_items_ranking_request(self, request: ItemsRankingRequest):
        self.items_ranking_requests.append(request)

    def get_item_rankings(self):
        raise NotImplementedError

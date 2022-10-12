from reasoning_path_utils import *


class ReasoningPath:
    def __init__(self, path, score, prob, lir, sep, path_type, pattern):
        self.path = path
        self.score = score
        self.prob = prob
        self.lir = lir
        self.sep = sep
        self.path_type = path_type
        self.pattern = pattern

    def len(self):
        len = 0
        for s in self.path:
            if type(s) != str:
                s = str(s)
            if s.isnumeric():
                len += 1
        return len


class TopkReasoningPaths:
    def __init__(self, dataset_name, paths):
        self.dataset_name = dataset_name
        self.topk = paths
        self.max_path_type = get_no_path_types_in_kg(dataset_name)

    # Time recency of linking interaction
    def topk_lir(self):
        lirs_topk = []
        for path in self.topk:
            lirs_topk.append(path.lir)
        return np.mean(lirs_topk)

    # Popularity of shared entity
    def topk_sep(self):
        seps_topk = []
        for path in self.topk:
            seps_topk.append(path.sep)
        return np.mean(seps_topk)

    # Diversity of linked interaction
    def topk_lid(self):
        unique_linking_interaction = set()
        for path in self.topk:
            linked_interaction_id, _ = get_linked_interaction_triple(path)
            unique_linking_interaction.add(linked_interaction_id)
        return len(unique_linking_interaction) / self.k

    # Diversity of shared entities
    def topk_sed(self):
        unique_shared_entity = set()
        for path in self.topk:
            shared_entity_id, _ = get_shared_entity_tuple(path)
            unique_shared_entity.add(shared_entity_id)
        return len(unique_shared_entity) / self.k

    # Diversity of path type
    def topk_ptd(self):
        unique_path_type = set()
        for path in self.topk:
            unique_path_type.add(path.path_type)
        return len(unique_path_type) / max(self.k, self.max_path_type)

    def topk_ptc(self):
        def simpson_index(topk):
            n_path_for_patterns = {k: 0 for k in set(get_path_types_in_kg(self.dataset_name))}
            N = 0
            for path in topk:
                path_type = self.path_type
                n_path_for_patterns[path_type] += 1
                N += 1
            numerator = 0
            for path_type, n_path_type_ith in n_path_for_patterns.items():
                numerator += n_path_type_ith * (n_path_type_ith - 1)

            if N * (N - 1) == 0:
                return 0
            return 1 - (numerator / (N * (N - 1)))

        ETC = simpson_index(self.topk)
        return ETC

    # Diversity of linked interaction type
    def topk_litd(self):
        unique_linking_interaction_types = set()
        for path in self.topk:
            _, _, linked_interaction_type = get_linked_interaction_triple(path)
            unique_linking_interaction_types.add(linked_interaction_type)
        return len(unique_linking_interaction_types) / self.k

    # Diversity of shared entities type
    def topk_setd(self):
        unique_shared_entity_type = set()
        for path in self.topk:
            _, shared_entity_type = get_shared_entity_tuple(path)
            unique_shared_entity_type.add(shared_entity_type)
        return len(unique_shared_entity_type) / self.k

    """
     # (self_loop user 0) (watched movie 2408) (watched user 1953) (watched movie 277) #hop3
     # (self_loop user 0) (mention word 2408) (described_as product 1953) (self_loop product 1953) #hop2
     def template_single(self):
          path = self.path
          if path[0] == "self_loop":
               path = path[1:]

          path_length = path.len()
          if path_length == 3:
               path = path[:-1]
          path_elements = {}
          for i in range(1, len(path)):
               relation, entity_type, entity_id = path[i]
               path_elements[f"r_{i - 1}"] = relation
               path_elements[f"e_{i - 1}"] = entity2name[entity_type][
                    entity_id] if entity_type != "user" else f"user_{entity_id}"

          if path_length == 4:
               return f"{path_elements['e_4']} is recommend to you because you {path_elements[f'r_0']} " \
                      f"{path_elements[f'e_1']} also {path_elements[f'r_3']} by {path_elements[f'e_2']}"

          elif path_length(path) == 3:
               _, uid, rel_0, e_type_1, e_1, rel_1, _, pid = path
               return f"{path_elements['e_3']} is recommend to you because is {path_elements['r_2']} " \
                      f"with {path_elements['e_1']} that you previously {path_elements['r_0']}"
     """


def pathfy(dataset_name, uid_paths):
    LIR_matrix = load_LIR_matrix(dataset_name)
    SEP_matrix = load_SEP_matrix(dataset_name)
    topk_reasoning_paths = {}
    for uid, path_triplets in uid_paths.items():
        curr_reasoning_paths = []
        for path_triplet in path_triplets:
            score, prob, path = path_triplet
            linked_interaction_id, linked_interaction_rel = get_linked_interaction_triple(path)
            shared_entity_id, shared_entity_type = get_shared_entity_tuple(path)
            lir = LIR_matrix[linked_interaction_rel][linked_interaction_id]
            sep = SEP_matrix[shared_entity_type][shared_entity_id]
            path_type = get_path_type(path)
            pattern = get_path_pattern(path)
            curr_reasoning_paths.append(ReasoningPath(path, score, prob, lir, sep, path_type, pattern))
        topk_reasoning_paths[uid] = TopkReasoningPaths(dataset_name, curr_reasoning_paths)
    return topk_reasoning_paths

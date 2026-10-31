from hybrid_collection import HybridCollection


hybrid_collection = HybridCollection("natural-questions")


def print_search_results(search_results, score_key, n=10):
    if not search_results:
        return

    print(score_key, ":")
    top_search_results = sorted(search_results, key=lambda key: -search_results[key][score_key])

    for doc_id in top_search_results[:n]:
        print(search_results[doc_id])


question = "Where is a hamester"
search_results = hybrid_collection.search(question)

score_keys = ["cos_sim", "max_sim"]
for k in score_keys:
    print_search_results(search_results, score_key=k)


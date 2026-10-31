from datasets import load_dataset
from hybrid_collection import HybridCollection


hybrid_collection = HybridCollection("natural-questions")


def from_dataset(dataset_name, split):
    pandas_df = load_dataset(dataset_name)[split].to_pandas()

    for cnt, row in pandas_df.iterrows():
        if cnt == 100:
            break

        hybrid_collection.ingest(
            row['answer'],
            row['query']
        )


if __name__ == "__main__":
    from_dataset('sentence-transformers/natural-questions',
         'train')
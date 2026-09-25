"""
Multi-strategy blocking and candidate generation module.
Amazon ML Challenge 2026.
"""

from collections import defaultdict, Counter
import polars as pl
from .normalize import SUFFIX_REGEX, ADDR_STOP_REGEX


def generate_keys_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    Generate rich multi-strategy blocking keys for a DataFrame of business records.
    Runs entirely in Polars Rust engine for maximum performance.
    """
    cleaned = df.with_columns(
        pl.col("business_name").fill_null("").str.to_lowercase()
            .str.replace_all(r"\.(com|org|net|in|us|fr|gov|edu|biz|info).*", "")
            .str.replace_all(SUFFIX_REGEX, " ")
            .str.replace_all(r"[^\w\s]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("c_name"),
        pl.col("business_address").fill_null("").str.to_lowercase()
            .str.replace_all(ADDR_STOP_REGEX, " ")
            .str.replace_all(r"[^\w\s]", " ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
            .alias("c_addr"),
        pl.col("business_name").fill_null("").str.to_lowercase()
            .str.replace_all(r"\.(com|org|net|in|us|fr|gov|edu|biz|info).*", "")
            .str.replace_all(SUFFIX_REGEX, "")
            .str.replace_all(r"[^\w]", "")
            .str.strip_chars()
            .alias("comp_name"),
        pl.col("business_address").fill_null("").str.extract_all(r"\b\d{2,6}\b")
            .list.eval(pl.element().str.strip_chars_start("0"))
            .list.eval(pl.element().filter(pl.element().str.len_chars() >= 2))
            .alias("nums")
    ).with_columns(
        pl.col("c_name").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 2)).alias("n_words"),
        pl.col("c_addr").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3)).alias("a_words")
    )

    res = cleaned.with_columns(
        pl.when(pl.col("comp_name").str.len_chars() >= 5)
          .then(pl.concat_str([pl.lit("c_"), pl.col("comp_name").str.slice(0, 10)]))
          .otherwise(pl.lit(""))
          .alias("k_comp"),
        pl.col("n_words").list.get(0, null_on_oob=True).fill_null("").alias("nw0"),
        pl.col("n_words").list.get(1, null_on_oob=True).fill_null("").alias("nw1"),
        pl.col("n_words").list.get(2, null_on_oob=True).fill_null("").alias("nw2"),
        pl.col("nums").list.get(0, null_on_oob=True).fill_null("").alias("num0"),
        pl.col("nums").list.get(1, null_on_oob=True).fill_null("").alias("num1"),
        pl.col("a_words").list.get(0, null_on_oob=True).fill_null("").alias("aw0"),
        pl.col("a_words").list.get(1, null_on_oob=True).fill_null("").alias("aw1"),
        pl.col("a_words").list.get(-1, null_on_oob=True).fill_null("").alias("aw_last"),
    ).with_columns(
        # Name bigrams
        pl.when((pl.col("nw0") != "") & (pl.col("nw1") != ""))
          .then(pl.concat_str([pl.lit("n2_"), pl.col("nw0"), pl.lit("_"), pl.col("nw1")]))
          .otherwise(pl.lit(""))
          .alias("k_n2_01"),
        pl.when((pl.col("nw1") != "") & (pl.col("nw2") != ""))
          .then(pl.concat_str([pl.lit("n2_"), pl.col("nw1"), pl.lit("_"), pl.col("nw2")]))
          .otherwise(pl.lit(""))
          .alias("k_n2_12"),
        # Name prefix + house number
        pl.when((pl.col("nw0") != "") & (pl.col("num0") != ""))
          .then(pl.concat_str([pl.lit("nn_"), pl.col("nw0").str.slice(0, 4), pl.lit("_"), pl.col("num0")]))
          .otherwise(pl.lit(""))
          .alias("k_nn"),
        pl.when((pl.col("nw0") != "") & (pl.col("num1") != ""))
          .then(pl.concat_str([pl.lit("nn_"), pl.col("nw0").str.slice(0, 4), pl.lit("_"), pl.col("num1")]))
          .otherwise(pl.lit(""))
          .alias("k_nn1"),
        # Name prefix + address words
        pl.when((pl.col("nw0") != "") & (pl.col("aw0") != ""))
          .then(pl.concat_str([pl.lit("na_"), pl.col("nw0").str.slice(0, 4), pl.lit("_"), pl.col("aw0")]))
          .otherwise(pl.lit(""))
          .alias("k_na0"),
        pl.when((pl.col("nw0") != "") & (pl.col("aw_last") != ""))
          .then(pl.concat_str([pl.lit("na_"), pl.col("nw0").str.slice(0, 4), pl.lit("_"), pl.col("aw_last")]))
          .otherwise(pl.lit(""))
          .alias("k_na"),
        # Number + address words
        pl.when((pl.col("num0") != "") & (pl.col("aw0") != ""))
          .then(pl.concat_str([pl.lit("numa_"), pl.col("num0"), pl.lit("_"), pl.col("aw0")]))
          .otherwise(pl.lit(""))
          .alias("k_numa0"),
        pl.when((pl.col("num0") != "") & (pl.col("aw1") != ""))
          .then(pl.concat_str([pl.lit("numa_"), pl.col("num0"), pl.lit("_"), pl.col("aw1")]))
          .otherwise(pl.lit(""))
          .alias("k_numa1"),
        pl.when((pl.col("num0") != "") & (pl.col("aw_last") != ""))
          .then(pl.concat_str([pl.lit("numa_"), pl.col("num0"), pl.lit("_"), pl.col("aw_last")]))
          .otherwise(pl.lit(""))
          .alias("k_numa"),
        # Address numbers combination
        pl.when((pl.col("num0") != "") & (pl.col("num1") != ""))
          .then(pl.concat_str([pl.lit("num2_"), pl.col("num0"), pl.lit("_"), pl.col("num1")]))
          .otherwise(pl.lit(""))
          .alias("k_num2"),
        # Address words bigram
        pl.when((pl.col("aw0") != "") & (pl.col("aw1") != ""))
          .then(pl.concat_str([pl.lit("a2_"), pl.col("aw0"), pl.lit("_"), pl.col("aw1")]))
          .otherwise(pl.lit(""))
          .alias("k_a2"),
        # Significant single name words
        pl.when(pl.col("nw0") != "")
          .then(pl.concat_str([pl.lit("nw_"), pl.col("nw0")]))
          .otherwise(pl.lit(""))
          .alias("k_nw0"),
        pl.when(pl.col("nw1") != "")
          .then(pl.concat_str([pl.lit("nw_"), pl.col("nw1")]))
          .otherwise(pl.lit(""))
          .alias("k_nw1"),
        pl.when(pl.col("nw2") != "")
          .then(pl.concat_str([pl.lit("nw_"), pl.col("nw2")]))
          .otherwise(pl.lit(""))
          .alias("k_nw2"),
    )

    key_cols = [
        "k_comp", "k_n2_01", "k_n2_12", "k_nn", "k_nn1",
        "k_na0", "k_na", "k_numa0", "k_numa1", "k_numa",
        "k_num2", "k_a2", "k_nw0", "k_nw1", "k_nw2"
    ]
    return res.select(["entity_id", "country"] + key_cols)


def build_inverted_index(keys_df: pl.DataFrame, max_frequency: int = 1200) -> dict:
    """
    Build an in-memory inverted index: key -> list of target entity IDs.
    Filters out keys exceeding max_frequency to avoid explosive Cartesian products.
    """
    index = defaultdict(list)
    for row in keys_df.iter_rows():
        tid = row[0]
        for k in row[2:]:
            if k:
                index[k].append(tid)

    # Prune high frequency keys
    pruned_index = {k: v for k, v in index.items() if len(v) <= max_frequency}
    return pruned_index


def query_candidates_for_s1(s1_keys_df: pl.DataFrame, inverted_index: dict, top_k: int = 30):
    """
    Query candidate targets for each S1 entity using the inverted index.
    Returns: dict mapping s1_id -> list of (candidate_id, match_count)
    """
    candidates_by_s1 = {}
    for row in s1_keys_df.iter_rows():
        s1_id = row[0]
        counter = Counter()
        for k in row[2:]:
            if k and k in inverted_index:
                for tid in inverted_index[k]:
                    counter[tid] += 1
        
        top_candidates = counter.most_common(top_k)
        candidates_by_s1[s1_id] = top_candidates
        
    return candidates_by_s1

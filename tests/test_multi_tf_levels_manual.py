import pandas as pd
from analysis.multi_tf_key_levels import (
    SingleTFLevel,
    CompositeLevel,
    cluster_levels_with_hdbscan,
)

def debug_cluster_demo():
    levels = [
        # 一堆围绕 10 的关键位
        SingleTFLevel(price=9.9, kind="support",    freq="D",   strength_tf=2),
        SingleTFLevel(price=10.05, kind="support",  freq="60m", strength_tf=1),
        SingleTFLevel(price=10.1, kind="support",   freq="W",   strength_tf=3),

        # 一堆围绕 15 的关键位（压力）
        SingleTFLevel(price=14.9, kind="resistance", freq="D",   strength_tf=2),
        SingleTFLevel(price=15.0, kind="resistance", freq="60m", strength_tf=1),
        SingleTFLevel(price=15.1, kind="resistance", freq="W",   strength_tf=3),

        # 一个远离的点（噪声）
        SingleTFLevel(price=30.0, kind="resistance", freq="D",   strength_tf=1),
    ]

    comps = cluster_levels_with_hdbscan(
        levels,
        eps_pct=0.01,       # 1% 容差
        min_cluster_size=2, # 至少2个点成簇
        allow_noise_as_singleton=True,
    )

    print("cluster count:", len(comps))
    for c in comps:
        print(
            f"price={c.price:.2f}, kind={c.kind}, total_strength={c.total_strength:.2f}, "
            f"members={len(c.members)}, freqs={[m.freq for m in c.members]}"
        )

if __name__ == "__main__":
    debug_cluster_demo()

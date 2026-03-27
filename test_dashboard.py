from dashboard import compute_metrics

metrics = compute_metrics(42)
for agent, data in metrics.items():
    print(f"\n{agent}:")
    for diff, m in data.items():
        s = m['score']
        c = m['cleared']
        r = m['reward']
        w = m['avg_wait']
        if c is not None:
            print(f"  {diff:8}: score={s} | cleared={c} | reward={r} | wait={w}")
        else:
            print(f"  {diff:8}: score={s}")

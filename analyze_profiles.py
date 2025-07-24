import pstats

def analyze(filename):
    """Loads a pstats file and prints the top 10 bottlenecks."""
    p = pstats.Stats(filename)
    
    print(f"\n--- Analysis for: {filename} ---")
    
    print("\n--- Top 10 by TOTAL TIME (tottime) ---")
    # tottime is the best indicator of where the code itself is slow
    p.sort_stats(pstats.SortKey.TIME).print_stats(10)
    
    print("\n--- Top 10 by CUMULATIVE TIME (cumtime) ---")
    p.sort_stats(pstats.SortKey.CUMULATIVE).print_stats(10)


# Analyze both files
analyze('profile_original.pstats')
analyze('profile_parallel.pstats')
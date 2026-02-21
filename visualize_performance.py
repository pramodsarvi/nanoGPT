
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_static_results():
    # Performance data from benchmark
    results = {
        'Baseline': 133.11,
        'KV Cache': 294.20,
        'Triton + KV': 287.19,
        'Optimized Mode': 290.91
    }
    
    names = list(results.keys())
    values = list(results.values())
    
    # Professional, simple color palette (Shades of Blue/Slate)
    colors = ['#ced4da', '#4dabf7', '#228be6', '#1864ab'] 
    
    plt.figure(figsize=(10, 6))
    
    # Set clean white background style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    bars = plt.bar(names, values, color=colors, edgecolor='#495057', linewidth=0.8, width=0.6)
    
    # Simple, clear labels
    plt.ylabel('Throughput (tokens/sec)', fontsize=11, fontweight='500', color='#212529')
    plt.title('nanoGPT Inference Performance Comparison', fontsize=14, fontweight='bold', pad=20, color='#212529')
    
    # Customize grid
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add values on top of bars efficiently
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 3, f'{yval:.1f} t/s', 
                 ha='center', va='bottom', fontsize=10, color='#495057')
        
    # Calculate speedup for annotation
    speedup = results['Optimized Mode'] / results['Baseline']
    plt.figtext(0.5, 0.02, f"Result: {speedup:.1f}x total throughput increase over baseline", 
                ha='center', fontsize=11, fontstyle='italic', color='#495057')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('performance_comparison.png', dpi=300)
    print("Graph saved as 'performance_comparison.png'")

if __name__ == "__main__":
    plot_static_results()

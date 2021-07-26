from typing import List
import matplotlib.pyplot as plt

def plot_stocks(x_values, predicted_now: List[int], COMPANY) -> None:
    """
    Plots predicted stock value with previous values.
    """
    x_values = x_values.tolist()

    fig, ax = plt.subplots(figsize=(16,10))

    # prepare data
    new_x = [None]*(len(x_values)-1)
    new_x.append(x_values[-1])
    new_x.extend(predicted_now)

    # predicted
    ax.plot(new_x, 'o', color = '#F49948')
    ax.plot(new_x, color = '#F49948', label = 'Predicted value')

    # previous values
    ax.plot(x_values, 'o', color='#29526D')
    ax.plot(x_values, color = '#8DB0C7', label = 'Previous values')

    ax.legend()
    plt.legend(loc=2, prop={'size': 16})

    labels = [-50, -40, -30, -20, -10, 'TODAY', 10, 20]
    ax.set_xticklabels(labels)
    
    ax.set_title(f'Previous and predicted stock values of {COMPANY}', size = 16)
    ax.set_xlabel('Days', size = 16)
    ax.set_ylabel('Stock Value', size = 16)

    plt.grid(color='grey', linestyle='-', linewidth=0.25)

    plt.box(False)  # borderless

    plt.savefig(f"stock_{COMPANY}.png", dpi=150, bbox_inches='tight')
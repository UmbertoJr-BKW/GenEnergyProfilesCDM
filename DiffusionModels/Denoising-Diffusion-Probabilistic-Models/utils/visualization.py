import matplotlib.pyplot as plt
from IPython.display import clear_output

def time_series_plot(tensor, ax):
    # Convert tensor to numpy array  
    numpy_data = tensor.detach().cpu().numpy()  
    
    
    ax.xlim(scatter_range)
    ax.ylim(scatter_range)
    ax.rc('axes', unicode_minus=False)

    # This will create a line for each row in your data. If that's too many lines,  
    # you might need to select a subset of rows to plot.  
    for i in range(numpy_data.shape[0]):  
        ax.plot(numpy_data[i, :])  

def scatter(sample, only_final, scatter_range = [-10, 10]):
    clear_output()
    if only_final:        
        # Create a figure and a set of subplots  
        fig, ax = plt.subplots(figsize=(7, 7))  
        time_series_plot(sample, ax)
        plt.show()

    else:
        step_size = sample.size(0)
        fig, axs = plt.subplots(1, step_size, figsize=(step_size * 4, 4), constrained_layout = True)
        for i in range(step_size):
            scatter = sample[i].detach().cpu().numpy()
            scatter_x, scatter_y = scatter[:,0], scatter[:,1]
            axs[i].scatter(scatter_x, scatter_y, s=5)
            axs[i].set_xlim(scatter_range)
            axs[i].set_ylim(scatter_range)
        plt.show()


def update_plot(i, data, scat):
    scat.set_offsets(data[i].detach().cpu().numpy())
    return scat


def plot_next_batch(next_batch, alpha_value=0.3):
    # Create a figure and a set of subplots  
    fig, ax = plt.subplots()  
    # This will create a line for each row in the data.
    for i in range(next_batch.shape[0]):  
        ax.plot(next_batch[i, :], alpha=alpha_value)  
    plt.show()  

    
def plot_schedules(beta_t_cos, beta_t_lin, alpha_t_cos, alpha_t_lin, alphabar_t_cos, alphabar_t_lin):
    plt.figure(figsize=(9,3))  
    
    #, beta_t_mix, alpha_t_mix, alphabar_t_mix

    plt.subplot(1,3,1)  
    plt.plot(beta_t_cos, label='Cosine')  
    plt.plot(beta_t_lin, label='Linear')  
    #plt.plot(beta_t_mix, label='Mix')  
    plt.title('Beta')  
    plt.legend()  

    plt.subplot(1,3,2)  
    plt.plot(alpha_t_cos, label='Cosine')  
    plt.plot(alpha_t_lin, label='Linear')  
    #plt.plot(alpha_t_mix, label='Mix')  
    plt.title('Alpha')  
    plt.legend()  

    plt.subplot(1,3,3)  
    plt.plot(alphabar_t_cos, label='Cosine')  
    plt.plot(alphabar_t_lin, label='Linear')  
    #plt.plot(alphabar_t_mix, label='Mix')  
    plt.title('Alpha Bar')  
    plt.legend()  

    plt.tight_layout()  
    plt.show()  
    
    
def plot_schedule(beta_t, alpha_t, alphabar_t, approach="Mix"):
    plt.figure(figsize=(9,3))  
    
    #, beta_t_mix, alpha_t_mix, alphabar_t_mix

    plt.subplot(1,3,1)  
    plt.plot(beta_t, label=approach)  
    plt.title('Beta')  
    plt.legend()  

    plt.subplot(1,3,2)  
    plt.plot(alpha_t, label=approach)  
    plt.title('Alpha')  
    plt.legend()  

    plt.subplot(1,3,3)  
    plt.plot(alphabar_t, label=approach)  
    plt.title('Alpha Bar')  
    plt.legend()  

    plt.tight_layout()  
    plt.show()  
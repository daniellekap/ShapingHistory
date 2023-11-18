import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy.cluster.hierarchy import dendrogram, linkage

def generate_image_from_VAE(encoding_16, model):

        if isinstance(encoding_16, pd.Series):
            encoding_16 = encoding_16.values
        # Generate the latent vector from the mean of the period
        new_tab_long = model.fc3(torch.from_numpy(encoding_16.astype('float32')).unsqueeze(0))

        # Generate the image from the model
        with torch.no_grad():
            return model.decoder(new_tab_long).squeeze()


def generate_mean_tablet_plot(df, model, column_name, grid_title, save_path=None):
    """
    Visualize generated images in a grid layout with a big title above the grid and return a DataFrame
    with each period's mean encodings, the corresponding primary value, and the sample size per period.
    
    :param df: DataFrame containing encoding train data
    :param model: PyTorch model to use for generating images
    :param column_name: Primary column for looping (e.g., 'Period_Name')
    :param grid_title: The title to display above the grid
    :param save_path: Path to save the figure, if provided
    :return: DataFrame with mean encodings, primary values, and sample sizes per period
    """
    # Extract unique values from the primary column
    primary_values = df[column_name].unique()
    # Prepare the DataFrame to return
    mean_encodings_df = pd.DataFrame()

    # Determine the number of rows needed for the grid
    num_rows = len(primary_values) // 6 + (len(primary_values) % 6 > 0)

    # Set up the figure and the axes grid for visualization
    fig, axes = plt.subplots(nrows=num_rows, ncols=6, figsize=(20, num_rows * 3))
    fig.suptitle(grid_title, fontsize=20, fontweight='bold')

    # Iterate over the primary values to generate images and collect data
    for idx, primary_value in enumerate(primary_values):
        row, col = divmod(idx, 6)

        # Filter the dataframe for the current primary value
        filtered_df = df[df[column_name] == primary_value]
        period_mean = filtered_df.drop([column_name], axis=1).mean()
        
        # Calculate the sample size for the current period
        sample_size = filtered_df.shape[0]

        # Append the mean encoding, primary value, and sample size to the mean_encodings_df
        period_mean_df = pd.DataFrame(period_mean).transpose()
        period_mean_df[column_name] = primary_value  # Add the primary value column
        period_mean_df['Sample Size'] = sample_size  # Add the sample size column
        mean_encodings_df = pd.concat([mean_encodings_df, period_mean_df], ignore_index=True)

        generated_image = generate_image_from_VAE(period_mean, model)
        
        # Plot the image in the grid
        axes[row, col].imshow(generated_image.cpu().numpy(), cmap='gray')
        axes[row, col].set_title(f'{primary_value} (n={sample_size})', fontsize=10)
        axes[row, col].axis('off')

    # Hide any empty subplots if the number of primary values is not a multiple of 6
    for j in range(idx + 1, num_rows * 6):
        fig.delaxes(axes.flatten()[j])

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

    # Reorder the mean_encodings_df columns to have the primary value column first, followed by the sample size
    cols = [column_name, 'Sample Size'] + [col for col in mean_encodings_df.columns if col not in [column_name, 'Sample Size']]
    mean_encodings_df = mean_encodings_df[cols]

    return mean_encodings_df


def hierarchical_clustering_and_dendrogram(df, class_column, title = None ,save_path=None):
    """
    Perform hierarchical clustering on the images based on their mean encodings, plot the full dendrogram, 
    and optionally save it.

    :param df: DataFrame with mean encodings and primary values
    :param class_column: The name of the column with the class (label)
    :param save_path: Path to save the dendrogram image (optional)
    """
    # Extract features for clustering, which are all columns except the class column
    features = df.drop(columns=[class_column, "Sample Size"])
    X = features.values
    labels =  [f"{char} - {num} samples" for char, num in zip(df[class_column].values, df['Sample Size'].values)]
    # Perform hierarchical/agglomerative clustering
    Z = linkage(X, 'ward')

    # Plot the full dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram - ' + title, fontsize = 36)
    plt.xlabel('')
    plt.ylabel('Distance')

    dendrogram(
        Z,
        leaf_rotation=90.,
        leaf_font_size=12.,
        labels=labels,
        show_contracted=True,  # to get a distribution impression in truncated branches
    )

    # Save the dendrogram to a file if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
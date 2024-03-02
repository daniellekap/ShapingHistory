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
    num_images = len(primary_values)
    num_rows = num_images // 6 + (num_images % 6 > 0)
    num_cols = min(num_images, 6)  # Use 6 or the number of images, whichever is smaller

    # Set up the figure and the axes grid for visualization
    fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, num_rows * 3), squeeze=False)
    fig.suptitle(grid_title, fontsize=20, fontweight='bold')

    # Flatten the axes array for easy iteration
    axes_flat = axes.flatten()

    # Iterate over the primary values to generate images and collect data
    for idx, primary_value in enumerate(primary_values):
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

        # Generate the image from the model
        generated_image = generate_image_from_VAE(period_mean, model)

        # Plot the image in the grid
        ax = axes_flat[idx]
        ax.imshow(generated_image.cpu().numpy(), cmap='gray')
        ax.set_title(f'{primary_value} (n={sample_size})', fontsize=10)
        plt.tight_layout()
        ax.axis('off')

    # Hide any remaining unused subplots
    for ax in axes_flat[num_images:]:
        ax.axis('off')

    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')

    plt.show()
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
from math import ceil
from tqdm import tqdm
import os
import argparse
import imageio

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

def read_image(path:str)->np.ndarray:
    """
    Read an image from a path.
    """
    X = Image.open(path)
    X = np.array(X)
    X = np.mean(X, axis=-1)
    return X

def plot_image_lst(image_lst, ranks):
    """
    Plot a list of images.
    """
    plt.figure(figsize=(20, 20))
    plt.tight_layout()

    for i in range(len(image_lst)):
        plt.subplot(ceil(len(image_lst)/2), ceil(len(image_lst)/2), i+1)
        if i == 0:
            plt.title('Original')
        else:
            plt.title(f'R={ranks[i-1]} approx.')
        plt.imshow(image_lst[i], cmap='gray')
        plt.axis('off')
    plt.savefig('images/images.png')
    plt.show()

def plot_singular_values(s: np.ndarray):
    """
    Plot the singular values.
    """
    plt.figure(figsize=(20, 20))
    plt.tight_layout()

    plt.semilogy(s, '-o', label='Singular Values')
    plt.xlabel('Rank')
    plt.ylabel('Energy')
    plt.title('Singular Values')
    plt.savefig('images/singular_values.png')
    plt.show()

# Calculate the cumulative sum of the singular values and plot it.
def plot_cumulative_sum_singular(s: np.ndarray):
    """
    Plot the cumulative sum of the singular values.
    """
    plt.figure(figsize=(20, 20))
    plt.tight_layout()

    plt.plot(np.cumsum(s)/np.sum(s), '-o', label='Cumulative Sum')
    plt.xlabel('Rank')
    plt.ylabel('% STD. Variation')
    plt.title('Cumulative sum of singular values')
    plt.savefig('images/cumulative_sum_singular.png')
    plt.show()

def run(
    image_path:str, 
    plot_images:bool=True, 
    plot_singular:bool=True, 
    plot_cumulative_singular:bool=True,
    run_for_k_ranks = False,
    k_rank = 150
):
    """
    Run the SVD algorithm.
    args:
        image_path:str, Path to the image.
        plot_images:bool, Plot the images.
        plot_singular:bool, Plot the singular values.
        plot_cumulative_singular:bool, Plot the cumulative sum of the singular values.
        run_for_k_ranks:bool, Run the algorithm for a specific ranges of rank.
        k_rank:int, Range of rank to run the algorithm for.
    """
    assert os.path.exists(image_path), f'{image_path} does not exist.'
    assert image_path.endswith('.png') or image_path.endswith('.jpg'), f'{image_path} is not an image.'

    # Check for directories and create appropriate directories.
    image_name = image_path.split('/')[-1].split('.')[0]

    if not os.path.exists('images'):
        os.mkdir('images')
    if not os.path.exists('images/' + image_name):
        os.mkdir('images/' + image_name)
    
    # some constants
    font = ImageFont.truetype('.fonts/NunitoSans-BoldItalic.ttf', size=60)
    ranks = [1, 5, 10, 20, 50, 100,150]
    image_lst = []

    # read the image
    X = read_image(image_path)
    print(f'Image shape: {X.shape}\n')
    print('Running SVD...\n')

    # run the SVD algorithm
    U, s, V = np.linalg.svd(X, full_matrices=False) # Return the Economy SVD of a matrix.
    print(
        f'Left Singular Vectors shape: {U.shape}, Singular Values shape: {s.shape}, Right Singular Vectors shape: {V.shape}\n'
    )
    # Overwrite the ranks list to create a interpolated list of ranks.
    if run_for_k_ranks:
        plot_images = False
        plot_singular = False
        plot_cumulative_singular = False
        ranks = [k for k in range(2, k_rank+2, 2)]
        compressed_image = []
        for rank in tqdm(ranks, desc='Approximating image...'):
            X_approx = U[:, :rank] @ np.diag(s[:rank]) @ V[:rank, :]
            save_img = Image.fromarray(X_approx).convert('L')
            draw_img = ImageDraw.Draw(save_img)
            draw_img.text((20, 20), f'Rank {rank} approxiamtion', font=font, fill=(255))
            # save_img.save(f'images/{image_name}/{image_name}_rank{rank}_approximation.jpg')
            compressed_image.append(np.array(save_img).astype(np.uint8))
        print('Writing images to Video...\n')
        imageio.mimwrite(f'images/{image_name}/{image_name}_image_approximation.mp4', compressed_image)
        compressed_image = []
        print('Done!')

    if plot_images:
        image_lst.append(X)
        for rank in ranks:
            X_approx = U[:, :rank] @ np.diag(s[:rank]) @ V[:rank, :]
            image_lst.append(X_approx)
        plot_image_lst(image_lst, ranks)

    if plot_singular:
        plot_singular_values(s)
    
    if plot_cumulative_singular:
        plot_cumulative_sum_singular(s)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--image_path', type=str, required=True, help='Path to the image.')
    argparser.add_argument('--plot_images', type=int, default=1, help='Plot the all images.')
    argparser.add_argument('--plot_singular', type=int, default=1, help='Plot the singular values on logarithmic scale.')
    argparser.add_argument('--plot_cumulative_singular', type=int, default=1, help='Plot the cumulative sum of the singular values.')
    argparser.add_argument('--run_for_k_ranks', type=int, default=0, help='Run the algorithm for a specific range of rank.')
    argparser.add_argument('--k_rank', type=int, default=150, help='Range of rank to run the algorithm for.')
    args = argparser.parse_args()

    run(
        image_path=args.image_path,
        plot_images=bool(args.plot_images),
        plot_singular=bool(args.plot_singular),
        plot_cumulative_singular=bool(args.plot_cumulative_singular),
        run_for_k_ranks=bool(args.run_for_k_ranks),
        k_rank=args.k_rank
    )

if __name__=="__main__":
    main()

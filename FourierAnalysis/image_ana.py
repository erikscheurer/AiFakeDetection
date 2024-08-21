# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image

# def fourier_analysis(image_path):
#     # Load the image
#     image = Image.open(image_path).convert('L')  # Convert image to grayscale
#     image_array = np.array(image)

#     # Perform Fourier transform
#     f_transform = np.fft.fft2(image_array)
#     f_transform_shifted = np.fft.fftshift(f_transform)  # Shift zero frequency components to the center

#     # Compute magnitude spectrum
#     magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)  # Add 1 to avoid log(0)

#     # Plot original image
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.imshow(image_array, cmap='gray')
#     plt.title('Original Image')
#     plt.axis('off')

#     # Plot Fourier magnitude spectrum
#     plt.subplot(1, 2, 2)
#     plt.imshow(magnitude_spectrum, cmap='gray')
#     plt.title('Fourier Magnitude Spectrum')
#     plt.axis('off')

#     plt.show()

# # Example usage
# image_path = '/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/archive/imagenet_ai_0419_biggan/train/ai/000_biggan_00093.png'
# fourier_analysis(image_path)

# import numpy as np
# import matplotlib.pyplot as plt
# from PIL import Image
# import glob
# from tqdm import tqdm

# def fourier_analysis_multiple_images(image_paths):
#     all_magnitudes = []

#     # Process each image
#     for image_path in tqdm(image_paths):
#         # Load the image and convert to grayscale
#         image = Image.open(image_path).convert('L')
#         image_array = np.array(image)

#         # Perform Fourier transform
#         f_transform = np.fft.fft2(image_array)
#         f_transform_shifted = np.fft.fftshift(f_transform)

#         # Compute magnitude spectrum
#         magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)

#         # Collect magnitudes
#         all_magnitudes.append(magnitude_spectrum)

#     # Stack all magnitudes together
#     all_magnitudes = np.array(all_magnitudes)

#     # Compute 2D histogram
#     histogram, xedges, yedges = np.histogram2d(
#         np.ravel(all_magnitudes),
#         np.ravel(all_magnitudes),
#         bins=[100, 100]
#     )

#     # Plot 2D histogram
#     plt.figure(figsize=(10, 8))
#     plt.imshow(histogram.T, origin='lower', cmap='hot', interpolation='nearest')
#     plt.title('2D Histogram of Fourier Magnitudes')
#     plt.xlabel('Frequency Component 1')
#     plt.ylabel('Frequency Component 2')
#     plt.colorbar(label='Counts')
#     plt.show()

# # Example usage
# image_folder = '/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/archive/imagenet_ai_0419_vqdm/train/ai/*.png'
# image_paths = glob.glob(image_folder)
# fourier_analysis_multiple_images(image_paths)


import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift
import random
import glob
from tqdm import tqdm
from PIL import Image


# Simulating image loading (Replace this with actual image loading)
def load_images(image_folder_path, num_images):
    images = np.zeros((num_images, 128, 128))
    image_paths = glob.glob(image_folder_path)

    for i in range(num_images):
        if i < len(image_paths):
            img = Image.open(image_paths[i]).convert(
                "L"
            )  # Open image and convert to grayscale
            img_resized = img.resize((128, 128))  # Resize image to 128x128
            images[i] = np.array(
                img_resized
            )  # Convert image to numpy array and store in images array

    return images


# Compute noise residuals (example function, replace with actual computation)
def compute_noise_residuals(images):
    # Placeholder for noise residuals computation
    mean = images.mean(axis=(1, 2), keepdims=True)
    noise_residuals = images - images.mean(axis=(1, 2), keepdims=True)
    return noise_residuals


def average_plus_fft(images):
    residuals = compute_noise_residuals(images)
    avg_residual = np.mean(residuals, axis=0)
    dft_result = fftshift(fft2(avg_residual))
    return np.log(np.abs(dft_result) + 1)


def std_plus_fft(images):
    residuals = compute_noise_residuals(images)
    avg_residual = np.std(residuals, axis=0)
    dft_result = fftshift(fft2(avg_residual))
    return np.log(np.abs(dft_result) + 1)


def fft_plus_average(images):
    images_fft = [fftshift(fft2(image)) for image in images]
    mean_images_fft = np.mean(images_fft, axis=0)
    return np.log(np.abs(mean_images_fft) + 1)


def fft_plus_std(images):
    images_fft = [fftshift(fft2(image)) for image in images]
    mean_images_fft = np.std(images_fft, axis=0)
    return np.log(np.abs(mean_images_fft) + 1)


# Perform analysis
def analyze_spectral_spectra(
    real_images,
    biggan_images,
    vqdm_images,
    sdv5_images,
    wukong_images,
    adm_images,
    glide_images,
    midjourney_images,
):

    analyzeation_function = fft_plus_std

    real_spectra = analyzeation_function(real_images)
    biggan_spectra = analyzeation_function(biggan_images)
    vqdm_spectra = analyzeation_function(vqdm_images)
    sdv5_spectra = analyzeation_function(sdv5_images)
    wukong_spectra = analyzeation_function(wukong_images)
    adm_spectra = analyzeation_function(adm_images)
    glide_spectra = analyzeation_function(glide_images)
    midjourney_spectra = analyzeation_function(midjourney_images)

    images = [
        real_spectra,
        biggan_spectra,
        vqdm_spectra,
        sdv5_spectra,
        wukong_spectra,
        adm_spectra,
        glide_spectra,
        midjourney_spectra,
    ]
    titles = [
        "ImageNet Spectra",
        "BigGan Spectra",
        "VQDM Spectra",
        "SDV5 Spectra",
        "Wukong Spectra",
        "ADM Spectra",
        "Glide Spectra",
        "Midjourney",
    ]

    fig, axs = plt.subplots(2, 4, figsize=(15, 8))

    iterator = 0
    for i in range(2):
        for j in range(4):
            im = axs[i, j].imshow(images[iterator], cmap="winter")
            axs[i, j].set_title(titles[iterator])
            axs[i, j].axis("off")
            # axs[i, j].set_xticklabels([])
            # axs[i, j].set_yticklabels([])
            fig.colorbar(im, ax=axs[i, j])
            iterator += 1

    fig.suptitle("Spectra Comparison", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(
        "/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/fft_plus_std.svg"
    )
    plt.show()


# Main function to run the analysis
def main():
    num_images = 2000

    # Load images (Replace these calls with actual data loading)
    real_images = load_images(
        "/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/archive/imagenet_ai_0419_biggan/train/nature/*.JPEG",
        num_images,
    )
    biggan_images = load_images(
        "/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/archive/imagenet_ai_0419_biggan/train/ai/*.png",
        num_images,
    )
    vqdm_images = load_images(
        "/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/archive/imagenet_ai_0419_vqdm/train/ai/*.png",
        num_images,
    )
    sdv5_images = load_images(
        "/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/archive/imagenet_ai_0424_sdv5/train/ai/*.png",
        num_images,
    )
    wukong_images = load_images(
        "/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/archive/imagenet_ai_0424_wukong/train/ai/*.png",
        num_images,
    )
    adm_images = load_images(
        "/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/archive/imagenet_ai_0508_adm/train/ai/*.PNG",
        num_images,
    )
    glide_images = load_images(
        "/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/archive/imagenet_glide/train/ai/*.png",
        num_images,
    )
    midjourney_images = load_images(
        "/home/lukas/Documents/SimTech/Semester3/Automotive_CV/Image_analysis/archive/imagenet_midjourney/train/ai/*.png",
        num_images,
    )

    analyze_spectral_spectra(
        real_images,
        biggan_images,
        vqdm_images,
        sdv5_images,
        wukong_images,
        adm_images,
        glide_images,
        midjourney_images,
    )


if __name__ == "__main__":
    main()

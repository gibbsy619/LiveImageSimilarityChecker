# Example: python image_hash.py -i images/input_images -o images/similar_images -s 16 -t 33

import cv2
import os
from pathlib import Path
import argparse

ap = argparse.ArgumentParser()

print("Current Working Directory:", os.getcwd())

ap.add_argument("-i", "--input_folder", type=str, required=True,
                help="folder to clean similar images")
ap.add_argument("-o", "--similar_images", type=str, required=True,
                help="folder to move similar images")
ap.add_argument("-s", "--hash_size", type=int, default=16,
                help="hash size")
ap.add_argument("-t", "--threshold", type=int, required=True,
                help="threshold for detecting similar images")

args = vars(ap.parse_args())

import cv2

def calculate_mean(pixels_list):
    """
    Calculate the mean value of a list of pixel values.

    Args:
        pixels_list (list): List of pixel values.

    Returns:
        float: Mean value of the pixel values.
    """
    mean = 0
    total_pixels = len(pixels_list)
    for i in range(total_pixels):
        mean += pixels_list[i] / total_pixels
    return mean

def grab_pixels(squeezed_frame):
    """
    Extract pixel values from a squeezed frame.

    Args:
        squeezed_frame (numpy.ndarray): Squeezed frame.

    Returns:
        list: List of pixel values.
    """
    pixels_list = []
    for x in range(0, squeezed_frame.shape[1], 1):
        for y in range(0, squeezed_frame.shape[0], 1):
            pixel_color = squeezed_frame[x, y]
            pixels_list.append(pixel_color)
    return pixels_list

def make_bits_list(mean, pixels_list):
    """
    Create a list of bits based on pixel values and mean.

    Args:
        mean (float): Mean value.
        pixels_list (list): List of pixel values.

    Returns:
        list: List of bits (255 or 0) based on pixel values and mean.
    """
    bits_list = []
    for i in range(len(pixels_list)):
        if pixels_list[i] >= mean:
            bits_list.append(255)
        else:
            bits_list.append(0)
    return bits_list

def hashify(squeezed_frame, bits_list):
    """
    Apply bit values to a squeezed frame to create a hashed frame.

    Args:
        squeezed_frame (numpy.ndarray): Squeezed frame.
        bits_list (list): List of bits.

    Returns:
        numpy.ndarray: Hashed frame.
    """
    bit_index = 0
    hashed_frame = squeezed_frame.copy()  # Make a copy to avoid modifying the original frame
    for x in range(0, squeezed_frame.shape[1], 1):
        for y in range(0, squeezed_frame.shape[0], 1):
            hashed_frame[x, y] = bits_list[bit_index]
            bit_index += 1
    return hashed_frame

def generate_hash(frame, hash_size):
    """
    Generate perceptual hash for a given frame.

    Args:
        frame (numpy.ndarray): Input frame.
        hash_size (int): Desired size of the perceptual hash.

    Returns:
        tuple: Tuple containing list of bits and the hashed frame.
    """
    frame_squeezed = cv2.resize(frame, (hash_size, hash_size))
    frame_squeezed = cv2.cvtColor(frame_squeezed, cv2.COLOR_BGR2GRAY)
    pixels_list = grab_pixels(frame_squeezed)
    mean_color = calculate_mean(pixels_list)
    bits_list = make_bits_list(mean_color, pixels_list)
    hashed_frame = hashify(frame_squeezed, bits_list)
    hashed_frame = cv2.cvtColor(hashed_frame, cv2.COLOR_GRAY2BGR)
    return bits_list, hashed_frame


def clean_folder(input_folder, similar_images, hash_size, threshold):
    files = (os.listdir(input_folder))
    list_length = len(files)
    i = 0
    k = 1
    frame = None
    hashed_frame = None
    duplicate_count = 0
    bits_list = []

    while i < len(files):
        sum_diff = 0

        if files[i] is not None:
            frame = cv2.imread(f"{input_folder}/{files[i]}")
            bits_list, hashed_frame = generate_hash(frame, hash_size)

        while k < len(files):
            if (i != k) and (files[k] is not None):
                new_frame = cv2.imread(f"{input_folder}/{files[k]}")
                new_bits_list, hashed_second_frame = generate_hash(new_frame, hash_size)

                for j in range(len(bits_list)):
                    if bits_list[j] != new_bits_list[j]:
                        sum_diff += 1

                print(f"{files[i]} -> {files[k]} sum_diff = {sum_diff}")

                im_h = cv2.hconcat([cv2.resize(frame, (450, 450)), cv2.resize(new_frame, (450, 450))])
                im_h2 = cv2.hconcat([cv2.resize(hashed_frame, (450, 450)), cv2.resize(hashed_second_frame, (450, 450))])
                im_v = cv2.vconcat([im_h, im_h2])

                if sum_diff <= hash_size * hash_size * threshold / 100:
                    Path(f"{input_folder}/{files[k]}").rename(f"{similar_images}/{files[k]}")
                    print(f"Deleted {k} element ({files[k]}) of {list_length}")
                    del files[k]
                    duplicate_count += 1
                else:
                    k += 1

                cv2.putText(im_v, f"SIMILAR: {duplicate_count}", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)
                cv2.imshow("Seeking for similar images...", im_v)
                cv2.waitKey(1)
                sum_diff = 0
        i += 1
        k = i + 1

# clean_folder(args['input_folder'], args['similar_images'], args['hash_size'], args['threshold'])

import os
import cv2

def webcam_comparison_to_folder(input_folder, similar_images, hash_size, threshold):
    """
    Compares live webcam images to images in a specified folder to detect similarities.
    Uses perceptual hashing to compare images efficiently.
    Stores similar images in a designated "similar_images" folder for further analysis.

    Args:
        input_folder (str): Path to the folder containing images to compare against.
        similar_images (str): Path to the folder where similar images will be stored.
        hash_size (int): Desired size of the perceptual hash (affects sensitivity).
        threshold (int): Percentage of hash bits that can differ for images to be considered similar.
    """

    # Get list of files in the input folder
    files = os.listdir(input_folder)
    
    # Access the webcam
    cap = cv2.VideoCapture(0)
    
    i = 0
    frame = None
    hashed_frame = None
    duplicate_count = 0
    bits_list = []

    while True:
        # Capture a frame from the webcam
        ret, new_frame = cap.read()

        if not ret:
            print("Error: Unable to capture frame from webcam")
            break
        else:
            sum_diff = 0

            # Read an image from the folder for comparison
            if files[i] is not None:
                frame = cv2.imread(f"{input_folder}/{files[i]}")
                
                # Generate perceptual hash for the images
                bits_list, hashed_frame = generate_hash(frame, hash_size)
                new_bits_list, hashed_second_frame = generate_hash(new_frame, hash_size)

                # Compare the hash bits of the two images
                for j in range(len(bits_list)):
                    if bits_list[j] != new_bits_list[j]:
                        sum_diff += 1

                print(f"{files[i]} -> new sum_diff = {sum_diff}")

                # Concatenate original and new images for display
                im_h = cv2.hconcat([cv2.resize(frame, (450, 450)), cv2.resize(new_frame, (450, 450))])
                im_h2 = cv2.hconcat([cv2.resize(hashed_frame, (450, 450)), cv2.resize(hashed_second_frame, (450, 450))])
                im_v = cv2.vconcat([im_h, im_h2])

                # Check if the images are similar based on the threshold
                if sum_diff <= hash_size * hash_size * threshold / 100:
                    # Create a new directory for storing similar images
                    directory = files[i][:7] + str(i)
                    parent_dir = similar_images
                    path = os.path.join(parent_dir, directory)
                    
                    # Create the directory if it doesn't exist
                    if not os.path.exists(path):
                        os.mkdir(path)
                        print("Directory '%s' created" % directory)
                    
                    # Save the similar and original images in the new directory
                    final_path = os.path.join(path, files[i])
                    original_final_path = os.path.join(path, 'original' + files[i])
                    cv2.imwrite(final_path, new_frame)
                    cv2.imwrite(original_final_path, frame)
                    
                    duplicate_count += 1

                # Display the count of similar images
                cv2.putText(im_v, f"SIMILAR: {duplicate_count}", (5, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)
                cv2.imshow("Seeking for similar images...", im_v)
                cv2.waitKey(1)
                sum_diff = 0
            i += 1

            # Reset index if reached the end of the list
            if i == len(files):
                i = 0


webcam_comparison_to_folder(args['input_folder'], args['similar_images'], args['hash_size'], args['threshold'])
import os
import cv2
import numpy as np
from collections import defaultdict
import time

# Define constants for window size and sliding step
WINDOW_SIZE = 8
RESIZED_SIZE = 4
SLIDING_STEP = 4


def process_image(image_path):
    # Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None or image.shape[0] <= WINDOW_SIZE or image.shape[1] <= WINDOW_SIZE:
        return []

    height, width = image.shape
    patterns = []

    # Slide a WINDOW_SIZE x WINDOW_SIZE window over the image
    for y in range(0, height - WINDOW_SIZE + 1, SLIDING_STEP):
        for x in range(0, width - WINDOW_SIZE + 1, SLIDING_STEP):
            window = image[y:y + WINDOW_SIZE, x:x + WINDOW_SIZE]
            # Resize to RESIZED_SIZE x RESIZED_SIZE
            small_window = cv2.resize(window, (RESIZED_SIZE, RESIZED_SIZE), interpolation=cv2.INTER_AREA)
            # Binarize the RESIZED_SIZE x RESIZED_SIZE window
            _, binary_window = cv2.threshold(small_window, 128, 1, cv2.THRESH_BINARY)
            # Convert to a 16-bit number
            bit_str = ''.join(str(int(binary_window[i, j])) for i in range(RESIZED_SIZE) for j in range(RESIZED_SIZE))
            pattern = int(bit_str, 2)
            patterns.append(pattern)

    return patterns


def traverse_directory(directory):
    pattern_counts = defaultdict(int)
    total_patterns = 0

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                image_path = os.path.join(root, file)
                patterns = process_image(image_path)
                for pattern in patterns:
                    pattern_counts[pattern] += 1
                    total_patterns += 1

    return pattern_counts, total_patterns


def main():
    directory = '101_ObjectCategories'
    start_time = time.time()
    pattern_counts, total_patterns = traverse_directory(directory)
    end_time = time.time()

    # Print the total number of patterns and their frequency distribution
    print(f'Total number of patterns: {total_patterns}')
    print('Pattern frequency distribution (sorted by count):')
    # Sort the pattern_counts dictionary by value in descending order
    for pattern, count in sorted(pattern_counts.items(), key=lambda item: item[1], reverse=True):
        print(f'Pattern {pattern:04x}: {count} times')

    # Print the total time taken
    print(f'Total time taken: {end_time - start_time:.2f} seconds')


if __name__ == '__main__':
    main()


if __name__ == '__main__':
    main()

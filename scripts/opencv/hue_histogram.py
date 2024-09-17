import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb


def create_hue_histogram(img: np.ndarray, s_threshold: int = 30, v_threshold: int = 30, center_hue: int = -1) -> np.ndarray:
    """
    Create a meaningful hue histogram from an OpenCV image, excluding pixels with low saturation or value,
    with an option to center a specific hue.

    This function converts the input image to HSV color space, applies thresholds to exclude pixels
    with low saturation or value (which have meaningless hue), and then creates a histogram of the
    hue values for the remaining pixels.

    Args:
        img (np.ndarray): Input image in BGR format (OpenCV default).
        s_threshold (int, optional): Minimum saturation for a pixel to be included. Defaults to 30.
        v_threshold (int, optional): Minimum value for a pixel to be included. Defaults to 30.
        center_hue (int, optional): Hue value to center the histogram on. Range 0-179. 
                                    Default is -1 (no centering).        

    Returns:
        np.ndarray: The normalized hue histogram.    

    Note:
        - The function assumes the input image is in BGR format (OpenCV default).
        - The histogram is normalized to the range [0, 1].
        - Hue values in OpenCV range from 0 to 179 (for 8-bit images).
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Shift hue if center_hue is specified
    if 0 <= center_hue <= 179:
        h = (h - center_hue + 90) % 180    

    # Create a mask for meaningful colors
    mask = cv2.bitwise_and(cv2.inRange(s, s_threshold, 255), cv2.inRange(v, v_threshold, 255))
    
    # Calculate histogram for masked hue values
    hist = cv2.calcHist([h], [0], mask, [180], [0, 180])

    # Normalize the histogram
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)    
    
    return hist

def plot_hue_histogram(hist: np.ndarray, center_hue: int = -1):
    """
    Plot a hue histogram with a color chart along the x-axis.

    Args:
        hist (np.ndarray): The hue histogram to plot.
        center_hue (int, optional): The hue value the histogram is centered on. 
                                    Default is -1 (no centering).
    """
    fig, (ax_hist, ax_color) = plt.subplots(2, 1, figsize=(10, 6), 
                                            gridspec_kw={'height_ratios': [4, 1]}, 
                                            sharex=True)
    
    # Plot the histogram
    ax_hist.plot(hist.ravel())
    ax_hist.set_title('Hue Histogram')
    ax_hist.set_ylabel('Frequency')
    ax_hist.set_xlim([0, 180])
    
    # Create a color array for the colorbar
    hue_range = np.linspace(0, 1, 180)
    saturation = np.ones_like(hue_range)
    value = np.ones_like(hue_range)
    hsv_colors = np.stack((hue_range, saturation, value), axis=1)
    rgb_colors = hsv_to_rgb(hsv_colors)
    
    # Plot the color chart
    ax_color.imshow([rgb_colors], aspect='auto', extent=[0, 180, 0, 1])
    ax_color.set_yticks([])
    
    # Set x-axis label
    plt.xlabel('Hue Value' + (' (Shifted)' if 0 <= center_hue <= 179 else ''))
    
    plt.tight_layout()
    plt.show()

# # Example usage 1
# if __name__ == "__main__":
#     import os

#     image_file_name = "tetris_blocks.png"
#     image_file_name = os.path.join(os.path.dirname(__file__), image_file_name)

#     image = cv2.imread(image_file_name)
#     cv2.imshow("Image", image)
#     cv2.waitKey(0)

#     hist = create_hue_histogram(image)

#     plot_hue_histogram(hist, center_hue=-1)


# Example usage 2
if __name__ == "__main__":
    import os, glob

    data_path = r'C:\Users\scbry\OneDrive - HAN\data\EVML\gesture_data'

    # grab the list of images in our data directory
    print("[INFO] loading images...")
    p = os.path.sep.join([data_path, '**', '*.png'])

    file_list = [f for f in glob.iglob(p, recursive=True) if (os.path.isfile(f))]
    print("[INFO] images found: {}".format(len(file_list)))

    # loop over the image paths
    for filename in file_list:        
        image = cv2.imread(filename)
        cv2.imshow("Image", image)

        hist = create_hue_histogram(image)
        plot_hue_histogram(hist)

        k = cv2.waitKey(1000) & 0xFF

        # if the `q` key or ESC was pressed, break from the loop
        if k == ord("q") or k == 27:
            break 

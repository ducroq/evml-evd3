import Augmentor 

#setting up an Augmentor pipeline
total_images=5000

# https://augmentor.readthedocs.io/en/stable/code.html#Augmentor.Pipeline.Pipeline.resize
# Passing the path of the image directory
p = Augmentor.Pipeline(source_directory=r'dataset', output_directory=r'../dataset_aug')
# Defining augmentation parameters and generating samples
#p.rotate_without_crop(probability=0.7, max_left_rotation=20, max_right_rotation=20, expand=False)
p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)  # Rotate an image by an arbitrary amount.
p.flip_left_right(probability=0.4)                                  # Flip (mirror) the image along its horizontal axis, i.e. from left to right.
p.flip_top_bottom(probability=0.4)                                  # Flip (mirror) the image along its vertical axis, i.e. from top to bottom.
p.rotate_random_90(0.75)
p.shear(probability=0.5,max_shear_left=10,max_shear_right=10)       # Shear the image by a specified number of degrees.
p.scale(probability=0.5,scale_factor=1.2)                           # Scale (enlarge) an image, while maintaining its aspect ratio. This returns an image with larger dimensions than the original image.
p.skew(probability=0.1,magnitude=0.5)                               # Skew an image in a random direction, either left to right, top to bottom, or one of 8 corner directions.
p.random_color(probability=0.6,min_factor=0.6,max_factor=1.5)       # Random change saturation of an image.
p.random_brightness(probability=0.6,min_factor=0.6,max_factor=1.5)  # Random change brightness of an image.
p.random_contrast(probability=0.6,min_factor=0.6,max_factor=1.5)    # Random change image contrast.
p.resize(1, 244, 244, resample_filter="BICUBIC")

p.sample(total_images, multi_threaded=True)                                              # Generate x number of samples from the current pipeline.

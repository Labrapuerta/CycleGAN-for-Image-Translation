# CycleGAN-for-Image-Translation
CycleGAN for unpaired image-to-image translation in histopathology

## Overview
This repository contains the implementation of a [CycleGAN model](https://github.com/junyanz/CycleGAN?tab=readme-ov-file) designed to translate medical images from CT (Computed Tomography) to MRI (Magnetic Resonance Imaging) and vice versa. The ability to convert between these two imaging modalities has significant potential in the field of medical imaging, as it can enhance diagnostic accuracy, reduce patient exposure to radiation, and improve the accessibility of different imaging techniques.

## Dataset
The model was trained on a dataset from [Kaggle](https://www.kaggle.com/datasets/darren2020/ct-to-mri-cgan/data), which contains paired CT and MRI images. The dataset was preprocessed to ensure consistency and quality:

- Image Size: All images were cropped and resized to (360, 360, 3) to ensure uniform input size for the model.
Channel Adjustment: MRI images, originally in grayscale (1 channel), were converted to RGB format (3 channels) to match the format of CT images and facilitate better learning for the CycleGAN model.
Preprocessing
Before training, the dataset underwent the following preprocessing steps:

- Cropping and Resizing: Images were cropped to focus on the region of interest and resized to (360, 360, 3).
Channel Transformation: MRI images, initially single-channel, were transformed to three channels by duplicating the grayscale values across the RGB channels. This ensured compatibility with the CycleGAN architecture, which expects three-channel inputs.

## Model Architecture
- Improved Generator with Residual Blocks
The generator architecture consists of downsampling layers followed by residual blocks and upsampling layers. Residual blocks help in preserving the essential features of the input image while allowing the network to learn complex transformations.

- PatchGAN Discriminator
The discriminator uses a PatchGAN architecture, which classifies each patch of the image as real or fake. This allows the model to focus on high-frequency details and improves the quality of the generated images.

- (In progress) CycleGAN with Gradient Penalty 
The CycleGAN model is designed to enforce cycle consistency, ensuring that an image translated from CT to MRI and back to CT remains unchanged. Additionally, a gradient penalty term is included to stabilize the training process and enforce smoothness in the discriminator's output.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or new features.

## License
This project is licensed under the MIT License. See the LICENSE file for more details.

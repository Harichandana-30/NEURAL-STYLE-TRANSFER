COMPANY : CODTECH IT SOLUTIONS

NAME : SAMUDRALA LAKSHMI HARI CHANDANA

INTERN ID : CT06DM1148

DOMAIN : ARTIFICIAL INTELLIGENCE

DURATION : 6 WEEKS

MENTOR : NEELA SANTHOSH


# NEURAL-STYLE-TRANSFER

## About the Project

This Python-based neural style transfer project blends the **content of one image** with the **artistic style of another** using a pretrained deep learning model (VGG19). The project leverages **PyTorch** and **feature representations from convolutional neural networks (CNNs)** to perform artistic transformation. It was built as a hands-on implementation of core computer vision and deep learning concepts.

This project allows users to turn photographs into artwork by mimicking the brushstrokes, color palettes, and patterns of famous styles — all powered by neural networks.

## Features

- Uses a **pretrained VGG19 network** for feature extraction
- Combines content from one image with style from another
- Handles image loading, normalization, and preprocessing
- Supports GPU acceleration when available
- Saves the final stylized image locally
- Displays intermediate and final outputs using matplotlib

## Why This Was Built

The project was developed to explore the intersection of **art and artificial intelligence**. Neural Style Transfer (NST) is a well-known application of deep learning that creatively demonstrates how machines can understand and reinterpret visual patterns. This implementation provides practical insights into:

- CNN feature maps
- Transfer learning using pretrained models
- Custom loss functions (content and style)
- Optimization in PyTorch

It also serves as a foundation for more advanced computer vision experiments, such as fast neural style transfer or GAN-based image synthesis.

## Prerequisites

- Python 3.7 or higher

### Required Libraries

- torch
- torchvision
- matplotlib
- Pillow

Install all dependencies using the following command:

pip install -r requirements.txt

## output

<div align="center">

  <table>
    <tr>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/817d52ce-b3b0-493f-b543-da98f79f35e5" width="250"/><br>
        <strong>Content Image</strong>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/279988c1-3bba-47ac-baa9-14fca1f8890d" width="250"/><br>
        <strong>Style Image</strong>
      </td>
      <td align="center">
        <img src="https://github.com/user-attachments/assets/d58fcd00-4bf2-4ca6-89c9-30dd2ab90e1c" width="250"/><br>
        <strong>Stylized Image</strong>
      </td>
    </tr>
  </table>

</div>

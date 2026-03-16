# Seven-Segment Display OCR for Measurement Instruments
This project implements a Real-Time Computer Vision solution using Python and OpenCV to automate data reading from 7-segment LED/LCD displays. It was specifically designed to bridge the gap between legacy laboratory hardware and modern digital data acquisition systems.

Key Features
- Real-Time Processing: Low-latency detection using optimized OpenCV operations.
- Dynamic Calibration: Interactive Control Panel (Trackbars) to adjust thresholding, slant correction, and sensitivity on the fly.
- Robust Digit Recognition: Custom logic to handle thin characters (like the number "1") and decimal points using advanced padding techniques.
- Perspective & Slant Correction: Built-in affine transformations to compensate for camera angles.
- Geometry-Based Logic: High-efficiency detection using ROIs (Regions of Interest) instead of heavy Deep Learning models, making it suitable for edge devices.

Technical Problem Solving

During development, several engineering challenges were addressed:
- Aspect Ratio Stability: Implemented copyMakeBorder padding to maintain consistent digit proportions, preventing misclassification of narrow characters.
- Decimal Point Reliability: Developed a specialized contour filtering system based on area and vertical position to stabilize the decimal point recognition.
- Morphological Processing: Used custom dilation kernels to bridge gaps in fragmented LED segments common in older equipment.

How it Works

- Preprocessing: The frame is converted to grayscale (or color-specific channels) and binarized.
- Contour Detection: The system identifies external contours and filters them by height and area.
- ROI Normalization: Each digit is resized and padded to a standard 100x150 pixel window.
- Bit-Mapping: The system checks 7 specific coordinate points (segments) within the ROI. If the pixel density exceeds the sensitivity threshold, the "bit" is set to 1.
- Dictionary Lookup: The resulting bit-tuple is matched against a predefined map to return the corresponding digit.
  
![Demonstration](assets/gif.gif)

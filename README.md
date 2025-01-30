# Analysis of IQA metrics with Noisy References

## Overview
The main goal of this workflow is to assess which metrics fare best when a clean target is not available for comparison. To this end, we generate some fake scientific images (specifically ARPES) and apply different levels of Poisson noise on it.

Please take a look at [notebook](https://github.com/zain-sohail/noisy-iqa/blob/main/metric_comparison.ipynb) for the complete experiments. 

The image quality metrics assessed are SSIM, MS-SSIM, VSI, and FSIM.

## Installation

1. Clone repository:
```
git clone https://github.com/yourusername/noisy-iqa.git
cd noisy_iqa
```

2. Create a virtual environment:
```
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```
pip install -r requirements.txt
```

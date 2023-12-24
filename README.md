# Unlocking Efficiency - The Power of Vectorization

My presentation on the topic of **vectorization** delivered at [PyData Prague](https://pydata.cz/) on **29. 2. 2024**.

## Talk Abstract

> In this talk, we will dissect a relevant yet sometimes overlooked part of the skill set of ML engineers and data scientists - code vectorization, the cornerstone of modern numerical libraries. The aim is to underscore the importance of knowing when and how to effectively apply this technique to significantly boost performance. Furthermore, we will provide practical insights regarding implementation, delve into the pros and cons, and discuss the impacts on the codebase. The code examples, using primarily Python and NumPy, will emphasize semantic portability across different libraries and programming languages.

## Terminology

### Vectorization

- Vectorization is the optimization technique in numerical computing that involves rewriting code to efficiently utilize vector operations provided by hardware
- It entails restructuring algorithms to perform operations on entire arrays or matrices at once, allowing multiple instructions to be executed in parallel
- The key characteristic is the absence of explicit loops, indexing, or element-wise operations
- Focus on high-level, array-oriented operations
- Improved performance by leveraging the parallel processing capabilities of modern CPUs and GPUs

### Implicit Parallelism

Refers to the ability of a system or programming model to **automatically parallelize computations** without requiring explicit instructions from the programmer.

## Use Cases For Different Roles

### Ordinary Developers

- **Numerical computing and simulation**: performing element-wise operations on multi-dimensional data
- **Image processing** - libraries like [OpenCV](https://opencv.org/) or [scikit-image](https://scikit-image.org/) heavily exploit fast, vectorized operations on images
- **Game development** - physics simulation or graphics processing, e.g., updating multiple game objects simultaneously

### Data Scientists

- **Data cleaning and preprocessing** - normalizing features, filling missing values, applying custom functions along specific axes
- **Exploratory Data Analysis (EDA)** - requires computing various statistics and visualizing trends
- **Time series analysis** - rolling operations (in a sliding window fashion), e.g., moving averaging or other windowed functions

### Machine Learning Engineers

- **Feature engineering** - creating interaction terms or applying mathematical transformations to features
- **Model training and inference** - vectorized operations play a key role in accelerating these processes
- **Neural network computations** - deep machine learning heavily relies on vectorization, e.g., forward and backward passes

## References

Slides based on [revealjs](https://revealjs.com/) created using [vscode-reveal](https://github.com/evilz/vscode-reveal).

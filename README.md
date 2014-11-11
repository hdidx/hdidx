Indexing for High-Dimensional Data
==================================

## What is `HDIdx`?

`HDIdx` is a python package provides index structures for high-dimensional data.
The current version implements the `Product Quantization` described in [1].

## Installation

Run the following command:

```
[sudo] pip install hdidx
```

- *NOTE* As `HDIdx` using `k-means` in `Product Quantization`, you need to install python bindings of OpenCV.

## Examples

## Reference
```
[1] Jegou, Herve, Matthijs Douze, and Cordelia Schmid.
    "Product quantization for nearest neighbor search."
    Pattern Analysis and Machine Intelligence, IEEE Transactions on 33.1 (2011): 117-128.
```

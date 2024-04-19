#!/usr/bin/env python3

import json

import labelme
import numpy
import PIL.Image


print("==> Loading dogs.json")
json_data = json.load(open("dogs.json"))

print("==> Creating mask")
image_shape = (json_data["imageHeight"], json_data["imageWidth"])
mask = numpy.zeros(image_shape, dtype=bool)
for shape in json_data["shapes"]:
    mask |= labelme.utils.shape_to_mask(
        img_shape=image_shape, points=shape["points"], shape_type=shape["shape_type"]
    )

print("==> Exporting mask.jpg")
PIL.Image.fromarray(mask.astype(numpy.uint8) * 255).save("mask.jpg")

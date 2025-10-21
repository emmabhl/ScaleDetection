PROMPT_TEMPLATE = """
You are given a microscope image. Find the scale bar (if present) and the scale text that labels it
(e.g. '400 μm', '5 mm'). Then measure the scale bar length in pixels (e.g. 377.0) and convert it to 
millimeters. In addition, identify the specimen orientation in the image (Lateral, Dorsal, Gall, 
Ventral, Head, Wing, Genitalia, Thorax, Palp ventral, Eggs, Frontal, Dorso-lateral, Abdomen, Larva).
Respond with JSON EXACTLY in the following format:
{"scale_bar_found": true, "bbox": [x, y, w, h], "measured_scale_length": 377.0, "declared_scale_length": "400.0", "units": "μm", "pixel_to_mm_ratio": 0.0010610079575596818, "orientation": lateral}.
If no scale bar or text present, return:
{"scale_bar_found": false, "bbox": [], "measured_scale_length": null, "declared_scale_length": null, "units": null,"pixel_to_mm_ratio": null, "orientation": lateral}.
Only respond with the JSON formatted object, nothing else.
"""


# GIS Data Science Tutorials

Welcome to the GIS Data Science workspace. This repository contains projects and tutorials focused on geospatial analysis, applied computer vision, and other interesting concepts.

## Index of Projects

- [Tutorial 1: SQL Polygon Redrawing](#tutorial-1-sql-polygon-redrawing)
- [Tutorial 2: Utility Pole Object Detection & OCR](#tutorial-2-utility-pole-object-detection--ocr)

---

## Tutorial 1: SQL Polygon Redrawing

**Path:** [`tutorial_1/`](./tutorial_1/)

This project focuses on resolving overlapping geometries using SQL (PostGIS). It features queries for processing geospatial data (such as GeoPackages).

**Key Features:**
- Redrawing overlapping polygons by computing geometric differences (`ST_Difference`, `ST_Union`).
- Assigning priority dynamically by counting point features (`public.locations`) contained within each polygon (`ST_Within`).
- Generating random locations for spatial testing and simulation.
- Handling complex spatial topologies with `ST_MakeValid`, `ST_Intersects`, and `ST_CollectionExtract`.

## Tutorial 2: Utility Pole Object Detection & OCR

**Path:** [`tutorial_2/`](./tutorial_2/)

This computer vision project uses Python to detect utility poles and their ID tags from images, subsequently extracting the tag numbers using Optical Character Recognition (OCR).

**Key Features:**
- Employs zero-shot object detection (Grounding DINO) to identify bounding boxes for text prompts like "utility pole" and "pole tag".
- Automatically crops the identified pole tags for downstream processing.
- Runs OCR on the cropped tag images to read the text/numbers physically written on the utility poles.
- Generates visual results showing the original images annotated with bounding boxes, confidence scores, and the extracted OCR text.

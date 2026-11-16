# Document tools

The image provides PDF extraction and rendering, image and spreadsheet
processing, numerical tools, and OCR through Tesseract and OCRmyPDF.
Compared with the
[OfficeQA paper's baseline environment](https://arxiv.org/html/2603.08655v1#A5.SS3),
it omits the heavier OCR dependencies in favor of using the model's own
vision capabilities, and it adds numerical packages.

`PyPDF2` is retained alongside `pypdf` to match the paper's preinstalled imports.
`fpdf2`, `img2pdf`, `pikepdf`, and `lxml` are part of OCRmyPDF's dependency chain.
NumPy, pandas, SciPy, SymPy, and mpmath support numerical, symbolic, and
high-precision calculations. statsmodels supports regression, time-series
filtering and forecasting. Shapely supports geometric and page-layout calculations.

`apt.txt` and `requirements.in` list the selected packages. Architecture-specific
lock files pin the installed versions. The Containerfile checks Python imports,
system-tool presence, and Tesseract's English data offline as the task user.

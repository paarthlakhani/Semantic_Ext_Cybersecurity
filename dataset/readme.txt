=====================
MALWARETEXTDB DATASET
=====================

This folder contains the datasets that constitute MalwareTextDB.

plaintext/ 
- contains the plaintext files after the PDF reports are processed with PDFMiner

annotations/ 
- contains the plaintext files with XML tags denoting nonsentence sections such as headings and covers
- contains the annotations files (.ann) for each plaintext file; the positions of the annotations are based on character counts 

tokenized/
- contains the tokenized reports with POS tags from Stanford's POSTagger and incorporating the annotations in BIO format
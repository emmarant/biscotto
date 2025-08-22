21.08.2025

--v02
+ [X] Chose Solution 1 (at least for now)
+ [X] Remove old selection process. Reshuffle code sections
-- v01
+ [X] Model specs are now saved and read from an CSV file. (Assuming that the midap model list is static and not being updated.) CSV is loaded from same git repo as notebook at run time.
+ [X] expand model table columns
+ [X] sanity check for oversized table (multiple rows). Sanity check for imported packages


??.08.2025

-- v00
Adding model info and specs in the selection list --> users can get some useful info on each model and select based on that
Suggested two solutions so far
+ [X] Solution 1: DataTable (for listing information for each model) + ipywidnget (for selecting from a list menu)
+ [X] Solution 2: html-rendered dataframe with inplace checkboxes for selection (allows checkbox at same row of each table entry, and also allows for hyperlinks within the table for the docs column)

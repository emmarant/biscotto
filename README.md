**25.08.2025**

- v03
+ [X] CSV model table is downloaded localy in current colab session --> model list and specs read from it
+ [X] segmentation comparison plots now include overlay of raw with segmentation countours for each model
+ [X] a **midap_ext.py** file is downloaded localy in session and imported by default. Contains:
      + function for listing models and assisting in selection of models
      + function for drawing segmentation comparison plots (fully based on original midap functions with only some added tweaks)
      + function for drawing contours of segmented instances
      
      __All of the above are patched onto the midap SegmentationJupyter class__

**21.08.2025**

- v02
+ [X] Chose Solution 1 (at least for now)
+ [X] Remove old selection process. Reshuffle code sections
+ [X] New tweaked comparison function, based on the one from midap
+ [X] separate new functions (comparison plotting, model listing) from the jupyter notebook itself. --> Should they live in the same user-accesible and editable repo, or hide it to avoid easy/accidental editing from users?
      

- v01
+ [X] Model specs are now saved and read from an CSV file. (Assuming that the midap model list is static and not being updated.) CSV is loaded from same git repo as notebook at run time.
+ [X] expand model table columns
+ [X] sanity check for oversized table (multiple rows). Sanity check for imported packages


**??.08.2025**

- v00
  
Adding model info and specs in the selection list --> users can get some useful info on each model and select based on that
Suggested two solutions so far
+ [X] Solution 1: DataTable (for listing information for each model) + ipywidnget (for selecting from a list menu)
+ [X] Solution 2: html-rendered dataframe with inplace checkboxes for selection (allows checkbox at same row of each table entry, and also allows for hyperlinks within the table for the docs column)

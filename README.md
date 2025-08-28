**28.08.2025**

-v04

+ [X] add inference time (together with hardware info) as a choise criterion for the user. --> 
+ [X] extension of midap_ext.py with functions for running all models AND provide with inference time at completion of run
+ [X] some more helper functions in midap_ext.py for printing colad session HW info etc

**27.08.2025**
 - v04
  
🏁 **comparison subplot grid** fully updated and checked for multi-image performance. Happy with how it currently is. More tweaks possible (slider for contrast change, grayscale inversion option, etc)
+ [X] update buttons mouse-over text and looks (icons). Small tweaks in barplot. **Opinion**: barplot should be replaces by a two point plot or simply --> two numbers...

**26.08.2025**
- v04
+ [X] new plots added: segmentation overlap map

**25.08.2025**

- v03
+ [X] CSV model table is downloaded localy in current colab session --> model list and specs read from it
+ [X] segmentation comparison plots now include overlay of raw with segmentation countours for each model
+ [X] a **midap_ext.py** file is downloaded localy in session and imported by default. Contains:
      + function for listing models and assisting in selection of models
      + function for drawing segmentation comparison plots (fully based on original midap functions with only some added tweaks)
      + function for drawing contours of segmented instances
      
      __All of the above are patched onto the midap SegmentationJupyter class__

**24.08.2025**

-v02
+ [X] add overlay comparison plots: contours of segmented instances are overplotted on raw image
+ [X] consider extending midap with all the new functions that are being introduced: extra plots, model lists, etc

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

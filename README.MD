### Overview

This repository contains the code for a front end dashboard that visualizes graffiti occurence and income across the largest cities in the United States.

### Structure

- app.py: main application that contains front end structure and callbacks
- Data-Clean.ipynb: helper script thats ran once to aggregate the raw data files into a format appropriate for visualization

### Sample Output:

The dashboard contains several key components:

- filters for city, state, and household income
- KPI cards displaying number of images containing/not containing graffiti
- Map figure displaying graffiti rate across U.S. cities, selecting a single point will drill down to the image level
- Sorted bar chart by graffiti image count
- Income versus graffiti count scatterplot
  ![alt text](image.png)

Sooo - we have been working a lot on this

If you see any API-Keys (its my fault) please use them wisely (dont use them) 

**Model**

Anwarkh1/Skin_Cancer-Image_Classification


**Folder structure:**

|----- data: Dataset the Skin Cancer Model was trained on (https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)
|
|----- out-of-sample: We tested the HuggingFace Model with another dataset, to validate the implementation in this project
|
|----- scans: some pictures of moles to fille the database (if you are sensitive - dont open)
|
|------ src: is the folder where the actuall app.py is saved (we used: %%writefile src/app.py in the SkinCancer.ipynb file)
|
|------ diary.csv: database of all the information 
| 
|------ SkinPal.ipynb: code for the main dashboard 


**Running the Dashboard**

- Make sure to enter your HF-Keys 
- 
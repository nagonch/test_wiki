# Managing Large Datasets

# Where?
Large datasets should not go on the lab’s main google drive share, where space is limited. Rather it should go in the [RoboticImagingLargeDatasets ](https://drive.google.com/drive/u/0/folders/0AL5CKpWDLu-pUk9PVA)drive 

# Tarballing
Sending / receiving many small files to a google drive doesn’t always work well. As an alternative, consider creating a tarball of your files and sending that.

If you have many files, creating a compressed archive can be very slow. Instead create an uncompressed tarball. This is essentially a concatenation of files into one larger file, with no compression. It is fast to both create and unpack a tarball file, much faster than compressing / uncompressing a zip file.

Linux command line to create a tarball:

`tar cvf <tarball_filename.tar> <input files>`

e.g. 

`tar cvf mytar.tar *`

will add all files and subfolders from the active directory into the tar file mytar.tar, maintaining the directory structure of the source.

If a single tar file is very large (> a few gigs) consider breaking it into smaller tar files, e.g. one per sub-folder of the dataset.
# Documentation
## Readme
Someone unfamiliar with your work should be able to understand what’s in the dataset without having to download the whole thing. So create a readme file outside the tarballed data that explains what exactly is inside the dataset. Include a list in the readme saying what is in each top-level file and subfolder. 

In the readme also include an explanation of any non-obvious / non-standard file formats, and pointers to code and paper(s) associated with the dataset.
## Organisation 
Think like a prospective user of the dataset: if they want a particular subset of the data, do they know where to look? Do they know what is in each folder? Do they know where any calibration data and other metadata should be found?

If the list of top-level files and subfolders does not make it obvious how the dataset is organised, reconsider how you’ve selected the top-level file and folder structure.
# Formats
Use universal file formats wherever possible:
* For stacks of images, a folder of `.png` files is better than a proprietary .mat file. 
* For metadata a plaintext `.json `file is better than a binary metadata format that would require a special tool to read and interpret.
## “Raw” files and metadata
There is no universal raw file format. If your dataset contains raw files you must include detailed metadata describing the specific format of those files. Ideally include example code for loading the files correctly, e.g. in matlab or python.

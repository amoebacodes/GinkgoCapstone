The BeadFinder is a software package developed by MSc in Automated Science students at Carnegie Mellon University in collaboration with the automation and the NGS team at Ginkgo Bioworks. It includes three different computer vision algorithms that report the locations and total number of beads that have been dispense into a 384-well plate by the Echo dispenser. It also shows and/or saves a thresholded image (heatmap), highlighting the darker areas (suggests bead presence) on the plate. Therefore, it can serve as a quality control tool, informing automation technicians of successful bead dispenses.  

### Command Line Interface
#### Inputs:

- image_path: path to the image being analyzed
- algorithm_name: one of "adaptive_thresholding", "average_thresholding", "deep_learning" or "deep_learning_aug"
    - deep_learning_aug is the deep learning model trained using augmented data
    - deep learning is the deep learning model trained without using augmented data
- label: the name of the plate. Will be displayed on the heatmap
- image_registration: whether to perform image_registration. 
    - The reference image is src.jpeg in the package
- show_heatmap: if true, show heatmap to notebook output if running Jupyter Notebook; show pop-up window if running on the command line
- output_path: if specified (i.e. is not none), save heatmap to path. If show_heatmap is false, user has to specify the output_path as it needs to save the heatmap if it is not displayed. If show_heatmap is true and output_path is specified, the heatmap is both displayed and saved. If show_heatmap is true but output_path is not specified, the heatmap is only displayed.

#### Outputs:

- print to command line well_ids with beads (e.g. A1)
- print to command line coordinates (0-383) of wells with beads
- print to command line total number and percentage of wells with beads
- show and/or save heatmap

#### Demo:

Run the following on the command line

    python3 my/path/to/main.py --image_path='221019_122109_1043495.jpg' --algorithm_name=adaptive_thresholding --output_path=output/path/img.png --image_registration=True --show_heatmap=True


### Jupyter Notebook
The inputs and outputs are identical to the command line interface. Please see demo.ipynb for an example.

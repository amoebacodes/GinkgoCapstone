The BeadFinder is a software package developed by MSc in Automated Science students at Carnegie Mellon University in collaboration with the automation and the NGS teams at Ginkgo Bioworks. It includes three different computer vision algorithms that report the locations and total number of beads that have been dispense into a 384-well plate by the Echo dispenser. It also shows and/or saves a thresholded image (heatmap), highlighting the darker areas (suggests bead presence) on the plate. Therefore, it can serve as a quality control tool, informing automation technicians of successful bead dispenses.  

**Adaptive thresholding and the deep learning models with augmentation** have better performances out of the algorithms we have developed and tested (See [Bead Detection Algorithms: Results]). 

## How to use:
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


## Method Overview
### Preprocessing
1. Optional image registration to the source image included in the package. The source image is an example image provided by Ginkgo.
2. Crop out the background and rotate the plate image so that the wells can be cropped out by simply drawing a grid.
3. M3, M4, A12, A13, P12, P13, C21 wells are excluded at the current moment due to screws and plate dock visible in the background.
### Heatmap Generation
The 'heatmap' is a thresholded grayscale image of the whole plate, used in adaptive thresholding but is also visually informative. This is done through:
1. Converting the rotated plate image to grayscale
2. Gaussian blurring to reduce noise
3. Using adaptive thresholding to isolate out locally dark areas (foreground, suggests beads) from light areas (background)
### Bead Detection Algorithms: Overview
#### Average Thresholding
1. Convert the rotated plate image to grayscale
2. Draw a grid to separate out each well
3. For each well, calculate the standard deviation, mean, and minimum of the intensities
4. If the minimum is more than 4 standard deviations away from the mean, then report bead found.
#### Adaptive Thresholding
1. Generate heatmap (see [Heatmap Generation])
2. Draw a grid to separate out each well from the heatmap
3. Focus on the center of the well, i.e. crop out the rim
4. For each center of the well, if there is only one value (background), then there are no beads; if there are two values (background and foreground), then report bead found.
#### Deep Learning
1. Draw a grid to separate out the wells from the RGB rotated plate image
2. Perform inference on each well image using the deep learning model either trained with augmented data or without augmented data. The models are trained for 5000 iterations on (before augmentation 420732) well images, with 18850 well images held out for validation and equal amount for the test set.
### Bead Detection Algorithms: Results
<table>
  <thead>
    <tr>
      <th>Algorithm Name</th>
      <th>Precision</th>
      <th>Recall</th>
      <th>F1-score</th>
    </tr>
  </thead>
  <tfoot>
    <tr>
      <td>Average Thresholding; on all wells</td>
      <td>0.8690</td>
      <td>0.5876</td>
      <td>0.7406</td>
    </tr>
  </tfoot>
  <tbody>
    <tr>
      <td>Adaptive Thresholding; on all wells</td>
      <td>0.9968</td>
      <td>0.9625</td>
      <td>0.9793</td>
    </tr>
    <tr>
      <td>Deep Learning without Augmentation; on test set</td>
      <td>0.9516</td>
      <td>0.9903</td>
      <td>0.9706</td>
    </tr>
    <tr>
      <td>Deep Learning with Augmentation; on test set</td>
      <td>0.9694</td>
      <td>0.9871</td>
      <td>0.9781</td>
    </tr>
  </tbody>
</table>
YOLO Explanations: CRP and PCX Generation Guide
This guide outlines the step-by-step process for updating YOLO model weights, setting up dataset paths, and generating Concept Relevance Propagation (CRP) and PCX analyses and plots.

1. Update Dataset and Weight Paths
First, you need to point the notebooks to your new data and model weights.

In yolo_crp_organized.ipynb: * Locate the "Load dataset and model" section.

Update the root_dir variable to point to your new dataset's root directory.

Update the ckpt_path variable to point to your new YOLO .pt weights file.

In yolo_pcx_organized.ipynb:

Under "Dataset loading", update root_dir to your new data location.

Under "Model loading", update ckpt_path to your new model weights.

2. Run Glocal Analysis (CRP)
Before extracting PCX or plotting, you need to generate the base CRP analysis files.

Open yolo_crp_organized.ipynb and locate the "Run analysis and visualize explanations" section.

Set the output_dir to where you want the CRP results saved.

Uncomment and execute the run_analysis(model_name, model, dataset, output_dir=output_dir, device=device) line. Note: This step can be slow as it processes the entire dataset.

3. Generate Reference Images
Reference images are required for both CRP and PCX concept visualizations.

In yolo_crp_organized.ipynb, find the "Generate / save reference images for concept prototypes" cell.

Update output_dir_crp, ref_imgs_path_12, and ref_imgs_path_6 to your desired output folders.

Define the layers_to_process and run the cell. This uses the get_ref_images function to calculate and save the reference concept images as .h5 files.

4. Extract PCX Attributions
Next, you need to extract the channel concept vectors for your detections.

Open yolo_pcx_organized.ipynb and go to the "Extract PCX attributions per detection and save artifacts" cell.

Update OUT_BASE to set the output directory for the PCX files.

Execute the cell. It will loop through the dataset, run detection forward passes, extract channel concept vectors for the target layers, and save them as .npy arrays alongside a meta_class_x.json metadata file.

5. Plot CRP Explanations
In yolo_crp_organized.ipynb, go to the "Plot explanations for a specific sample" section.

Configure the parameters: set the class_id, the sample_id (the specific image number in the dataset), and the target layer.

Run the plot_explanations(...) function to render the visualizations for that specific detection.

6. Plot PCX Explanations
You have three different methods to visualize PCX in yolo_pcx_organized.ipynb.

Configuration: For all methods, first update the n_prototypes_by_layer dictionary to specify the layers you want to plot and the number of prototypes per class.

Method A (Specific Image Path): Under section 6A, provide an exact image_path to an image file. The notebook will run predictions and call plot_pcx_explanations(...) to save plots to pcx_plots.

Method B (Validation Dataset Index): Under section 6B, set val_idx to pull a specific index from your validation dataset to visualize.

Method C (Entire Dataset Loop): Under section 6C, you can loop through the entire training dataset. It will process every image, find detections, and save PCX explanation grids automatically to the designated output_dir_pcx paths.
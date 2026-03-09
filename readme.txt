PYTHON SCRIPTS:
final-feature-extraction.py = contains r3d18 model and dcp as feature extractors saved into csv for training(r3d18 has 512 cols; dcp has 1; label has 1)
final-train-mlp-v2.py = using extracted features as input for training MLP (trained for 50 epochs)
final-real-time-test.py = placeholder for testing the model using cv2 for real time analysis
data-check.py = visualization of split statistics

    --- Rigorous Split Statistics (Sequential 80/20) ---
Class        | Train      | Val        | Total
--------------------------------------------------
Clear        | 5996       | 1500       | 7496
Light Rain   | 5996       | 1500       | 7496
Heavy Rain   | 5996       | 1500       | 7496
--------------------------------------------------
TOTAL        | 17988      | 4500       | 22488

Final Split: 80.0% Training | 20.0% Validation

FOLDERS:
models & utils = contains the backbone of the r3d18 model

TRAINED WEIGHTS WILL BE IN A GDRIVE FOLDER DUE TO STORAGE LIMITATIONS
r3d18_best_checkpoint_rigorous= contains the best fine tuned epoch of the r3d18 for feature extractor during inference (checkpoint_epoch_4.pth)
mlp_best_checkpoint_rigorous = contains best mlp model (rigorous_epoch_28.pth)
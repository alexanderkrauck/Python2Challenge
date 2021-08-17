supplements_ex5/example_testset.pkl : Example testset as pickle file.
It contains a dictionary with keys
"input_arrays" (list of "input_array" as created via ex4 for each sample),
"known_arrays" (list of "known_array" as created via ex4 for each sample),
"borders_x" ("border_x" kwarg as used for ex4 for each sample),
"borders_y" ("border_y" kwarg as used for ex4 for each sample),
"sample_ids" (sample ID as string for each sample).

supplements_ex5/example_targets.pkl : List of "target_array" as created via ex4 for each sample.
This is the solution to the example testset and will not be available for the real test set.

supplements_ex5/example_submission_random.pkl : List of random predictions for each sample.

supplements_ex5/example_submission_perfect.pkl : List of perfect predictions for each sample.
This is a perfect solution and therefore identical to /media/michael/HDProjects/temp/python_2021/supplements_ex5/example_targets.pkl.


supplements_ex5/scoring.py : Scoring function as used for the scoring on the real testset.
The higher the score, the better the predictions.
Random predictions achieve a score of -8335.495, perfect predictions achieve a score of 0.0 on the example testset.

supplements_ex5/plots : Some visualized examples for random predictions on the example testset.

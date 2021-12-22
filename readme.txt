Included are:
	DataPreperation.ipynb - turns simulation files ".sim" from MEGAlib into ".csv" files usable by the neural network
	PETTracker.ipynb      - all programs neural network related, contains the model, can load and save model, can create data sets and test them, also contains optuna training for hyperparameters

	Example Data        - example of the prepared data
	Example Simulation  - example of a simulation file
	Output              - the network will save its output in this folder, this was already done for the file in the "Example Data" folder
	Saved Network       - the trained network and it's parameters is saved in this folder
	TrainNorm           - the normalization values used for the training data are saved in this folder
	PETTracker-study.db - the database for the optuna hyperparameter tuning
	SimToDataframe.py   - functions required by "DataPreperation.ipynb"

The ".ipynb" are jupyter notebooks that include the main code to run the data preperation as well as the network.
Simulation data should first be prepared with "DataPreperation.ipynb". The resulting ".csv" files can be used with the neural network in "PETTracker.ipynb".
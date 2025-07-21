import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import shap
from SALib.analyze import sobol
from object_utils import gpc_multiindex2param_names
#from scipy.stats import sobol_indices

class DNNModel(nn.Module):
    def __init__(self, input_size, output_size, layers, outputAF):
        super(DNNModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        default_neurons = 32
        default_activation = 'relu'
        default_dropout = 0.0

        # Define a mapping from string names to PyTorch activation functions
        activation_mapping = {
            'gelu': nn.GELU(),
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leakyrelu': nn.LeakyReLU(),
            'softmax': nn.Softmax(dim=1),
            'none': nn.Identity()
        }
        
        self.outputAF = activation_mapping.get(outputAF.lower()) if outputAF else nn.Identity()

        hidden_layers = []
        prev_size = input_size

        # Build hidden layers
        for layer in layers:
            neurons = default_neurons if layer['neurons'] is None else layer['neurons']
            activation = default_activation if layer['activation'] is None else layer['activation']
            dropout = default_dropout if layer['dropout'] is None else layer['dropout']

            # Add Linear, Activation, and Dropout layers to the list
            hidden_layers.append(nn.Linear(prev_size, neurons))
            hidden_layers.append(activation_mapping.get(activation.lower()))
            hidden_layers.append(nn.Dropout(dropout))
            prev_size = neurons

        # Create Sequential model
        self.hidden_layers = nn.Sequential(*hidden_layers)
        self.output_layer = nn.Sequential(nn.Linear(prev_size, output_size), self.outputAF)
        
        self.all_layers = nn.Sequential(*hidden_layers, nn.Linear(prev_size, output_size), self.outputAF)
        
        self.to(self.device)

    def forward(self, x):
        # hid_x = self.hidden_layers(x)
        # out_x = self.output_layer(hid_x) OR
        
        out_x = self.all_layers(x)
        return out_x

    
    def train_and_validate(self, q_train, y_train, q_val, y_val, loss='mse', optimizer='adam', epochs=5, batch_size=64, early_stopping=None):
        self.batch_size = batch_size
        train_loader = DataLoader(TensorDataset(q_train, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(TensorDataset(q_val, y_val), batch_size=batch_size, shuffle=False)
        
        # Define a mapping from string names to PyTorch criterion functions
        criterion_mapping = {
            'mse': nn.MSELoss(),
            'mae': nn.L1Loss(),
            'crossentropy': nn.CrossEntropyLoss()
        }
        criterion = criterion_mapping.get(loss.lower())

        # Define a mapping from string names to PyTorch optimizer functions
        optimizer_mapping = {
            'adam': optim.Adam(self.parameters()),
            'RMSprop': optim.RMSprop(self.parameters())
        }
        optimizer = optimizer_mapping.get(optimizer.lower())
        
        # Early stopping initialization
        if early_stopping is not None:
            # Extract parameters with defaults
            patience = early_stopping.get('patience', 15)
            min_delta = early_stopping.get('min_delta', 0.0)
            monitor = early_stopping.get('monitor', 'val_loss')  # e.g., 'val_loss', 'val_accuracy'
            mode = early_stopping.get('mode', 'min')  # 'min' for loss, 'max' for accuracy

            # Initialize tracking variables based on mode
            best_metric = float('inf') if mode == 'min' else -float('inf')
            epochs_no_improve = 0
            early_stop = False
            best_model_state = None  # To store the best model's state

        for epoch in range(epochs+1):
            self.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()   
            avg_train_loss = running_loss / len(train_loader)

            # Validation phase
            self.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    outputs = self(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            
            # Calculate monitored metric (for simplicity, we only track val_loss here)
            monitored_value = avg_val_loss  # Replace with your metric (e.g., accuracy) if needed

            
            if epoch % 50 == 0:
                print(f"Epoch [{epoch}/{epochs}], train loss: {avg_train_loss:.4f}, validation loss: {avg_val_loss:.4f}")
            
            # Early stopping logic
            if early_stopping is not None:
                if mode == 'min':
                    improvement = (monitored_value < best_metric - min_delta)
                elif mode == 'max':
                    improvement = (monitored_value > best_metric + min_delta)
                else:
                    raise ValueError(f"Invalid mode: {mode}. Use 'min' or 'max'.")

                if improvement:
                    best_metric = monitored_value
                    epochs_no_improve = 0
                    best_model_state = self.state_dict().copy()  # Save best model
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f'Early stopping triggered after {epoch + 1} epochs!')
                        early_stop = True
                        break

        # Restore the best model if early stopping was triggered
        if early_stopping is not None and early_stop:
            print(f'Training stopped early. Restoring best model (monitored {monitor} = {best_metric:.4f}).')
            self.load_state_dict(best_model_state)
        else:
            print('Training completed without early stopping.')  
        return avg_train_loss, avg_val_loss

    def evaluate(self, test_dataset):
        self.eval()
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        predictions = []
        targets = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self(inputs)
                predictions.extend(outputs.cpu().numpy())
                targets.extend(labels.cpu().numpy())

        # Calculate the mean squared error for regression
        mse = mean_squared_error(targets, predictions)
        print(f"Mean Squared Error: {mse:.4f}")
        return mse
    
    def predict(self, q):
        self.eval()  # Set the model to evaluation mode
        test_loader = DataLoader(q, batch_size=64, shuffle=False)  # Create DataLoader for test data
        predictions = []  # List to store the predictions

        with torch.no_grad():  # Disable gradient calculation as we only need predictions
            for inputs in test_loader:  # We don't need the labels (targets), so we use "_" to ignore them
                inputs = inputs.to(self.device)
                outputs = self(inputs)  # Get the model's predictions (outputs)
                predictions.extend(outputs.cpu().numpy())  # Add the predictions to the list (move to CPU and convert to numpy)     
        predictions = np.array(predictions)

        return predictions  # Return the list of predictions
    
    def compute_partial_vars(self, model_obj, max_index):
        paramset = model_obj.Q
        QoI_names = model_obj.QoI_names
        # problem = {
        #     'num_vars': paramset.num_params(),
        #     'names': paramset.param_names(),
        #     'bounds': [[0, 1] for _ in range(paramset.num_params())]
        # }
        problem = {
            'num_vars': paramset.num_params(), 'names': paramset.param_names(), 'dists': paramset.dist_types, 'bounds': paramset.dist_params
            } 
        
        d = paramset.num_params()
        q = paramset.sample(method='Sobol_saltelli', n=8192) # saltelli working only for uniform distribution # N * (2D + 2)
        # https://salib.readthedocs.io/en/latest/user_guide/advanced.html
        #q = paramset.sample(method='QMC_LHS', n=10000)
        #xi = model_obj.get_scaled_q(q)      
        #y = model_obj.model.predict(q)
        y = model_obj.predict(q)
        #y = model_obj.get_orig_y(y_t)
        
        # Run model
        S1 = []
        S2 = []
        for i in range(y.shape[1]):
            y_i = y[:,i]

            # Sobol analysis
            Si_i = sobol.analyze(problem, y_i)
            T_Si, first_Si, (idx, second_Si) = sobol.Si_to_pandas_dict(Si_i)
            df = Si_i.to_df()
            cols_S1 = list(df[1].index)
            cols_S2 = list(df[2].index)

            S1.append(first_Si['S1'])
            S2.append(second_Si['S2'])

        S1 = np.array(S1)
        S2 = np.array(S2)

        col_names = cols_S1
        sobol_index = S1
        if max_index == 2:
            sobol_index = np.concatenate([S1, S2], axis=1)
            col_names = cols_S1 + cols_S2
            col_names = [f"{x[0]} {x[1]}" if isinstance(x, tuple) else x for x in col_names]
                    
        # Compute partial variances
        y_var = y.var(axis=0).reshape(-1, 1)
        partial_variance = sobol_index * y_var
             
        partial_var_df, sobol_index_df = pd.DataFrame(partial_variance, columns=col_names, index=QoI_names), pd.DataFrame(sobol_index, columns=col_names, index=QoI_names)

        return partial_var_df, sobol_index_df, y_var
    
    # def compute_partial_vars(self, model_obj, paramset, max_index):
    #     a = []
    #     sobol_indices(model_obj.model, 2500, )
    #     pass
    
    def get_shap_values(self, predict_fn, q, forced=False, explainer_type="kernelexplainer"):
        self.eval()
        if explainer_type == "deepexplainer":
            # Determine device
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.to(device)
            
            # Prepare input
            if isinstance(q, pd.DataFrame):
                xi = torch.from_numpy(q.values).double().to(device)
            elif isinstance(q, np.ndarray):
                xi = torch.from_numpy(q).double().to(device)
            elif isinstance(q, torch.Tensor):
                xi = q.double().to(device)
            else:
                raise ValueError(f"Input q must be a numpy array or a torch tensor. The type of q is: {type(q)}")
            
            #xi = torch.tensor(xi, dtype=torch.float64)
            if hasattr(self, 'explainer') == False or forced == True:
                print(type(xi))
                self.eval()
                explainer = shap.DeepExplainer(self, xi)
                self.explainer = explainer
                #shap_values = explainer.shap_values(xi)
            self.eval()
            shap_values = self.explainer(xi)
        elif explainer_type == "kernelexplainer":
            if hasattr(self, 'explainer') == False or forced == True:
                explainer = shap.KernelExplainer(predict_fn, q)
                self.explainer = explainer
            shap_values = self.explainer(q)
        return shap_values

# Example usage
if __name__ == "__main__":
    # EXAMPLE
    # Parameters
    input_size = 2
    output_size = 1   # Single continuous output value
    batch_size = 64
    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1
    total_samples = 10000  # Number of data samples
        
    # Layer configuration: specify neurons, activation, and dropout
    init_config = [
        (512, 'relu', 0.2),  # All parameters specified
        (256, 'sigmoid'),    # Only neurons and activation specified
        (128,),              # Only neurons specified
        ()                   # All parameters default
    ]

    # Initialize the model
    model = DNNModel(input_size, output_size, init_config)
    model = model.double()

    # Linear function to generate y values
    # y = 3*x + 2 + some noise

    # # Generate input data
    # X = torch.randn(total_samples, input_size)
    # true_weights = torch.randn(input_size, 1) * 0.1  # True weights
    # true_bias = 2.0  # True bias

    # # Generate output data with some noise
    # y = X @ true_weights + true_bias + 0.1 * torch.randn(total_samples, 1)

    # # Create dataset
    # dataset = TensorDataset(X, y)      

    # # Split dataset into train, and test sets
    # train_size = int(train_ratio * total_samples)
    # test_size = total_samples - train_size

    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    
    # Generate input data
    X = np.random.randn(total_samples, input_size).astype(np.float64)
    true_weights = (np.random.randn(input_size, 1) * 0.1).astype(np.float64)  # True weights
    true_bias = np.float64(2.0)  # True bias

    # Generate output data with some noise
    y = X @ true_weights + true_bias + (0.1 * np.random.randn(total_samples, 1)).astype(np.float64)

    # Create dataset
    # Split dataset into train, val, and test sets
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=train_size, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=val_size, random_state=42)

    model.train_model(X_train, y_train)
    # # Run `train_model` using cross-validation
    # #model.train_model(train_dataset, criterion='mse', optimizer='adam', num_epochs=10, k_fold=5, batch_size=64)

    # # # With 80-20 ratio
    # model.train_model(train_dataset[0], train_dataset[1])

    # # Evaluate the model
    # model.evaluate(test_dataset)
    
    # # Predict 
    # #model.predict(test_dataset)
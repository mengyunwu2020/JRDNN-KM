import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cluster  # Ensure this module and kmeans function are correctly defined.

class WeightedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(WeightedMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
        self.weight = nn.Parameter(torch.rand(input_dim))

    def forward(self, x):
        x = x * self.weight
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x

    def weight_penalty(self):
        return torch.sum(torch.abs(self.weight))

def expand_weight_matrices(class_weights):
    """Expands weight matrices from p*(p-1) to p*p with diagonal elements set to 1."""
    extended_class_weights = []
    for weights_matrix in class_weights:
        p = weights_matrix.shape[0]
        extended_matrix = np.eye(p)
        for i in range(p):
            insert_indices = list(range(p))
            del insert_indices[i]
            extended_matrix[i, insert_indices] = weights_matrix[i, :]
        extended_class_weights.append(extended_matrix)
    return extended_class_weights

n, p, k = 1000, 10, 5  # Number of samples, features, classes
X = np.random.rand(n, p)
labels = np.random.randint(0, k, size=n)

mlp_models = [WeightedMLP(p-1, 64, 1) for _ in range(k * p)]

optimizer = optim.Adam([param for model in mlp_models for param in model.parameters()], lr=0.01)
criterion = nn.MSELoss()

for epoch in range(100):
    total_loss = 0
    for i, model in enumerate(mlp_models):
        optimizer.zero_grad()
        class_label = i // p
        variable_index = i % p
        indices = np.where(labels == class_label)[0]
        data = X[indices]
        response = data[:, variable_index]
        predictors = np.delete(data, variable_index, axis=1)
        
        response_tensor = torch.tensor(response, dtype=torch.float32).unsqueeze(1)
        predictors_tensor = torch.tensor(predictors, dtype=torch.float32)
        
        output = model(predictors_tensor)
        loss = criterion(output, response_tensor)

        penalty1 = model.weight_penalty()
        penalty2 = 0  # Initialize penalty for weight differences
        penalty3 = torch.norm(model.fc1.weight, p=2)  # Group Lasso penalty
        
        if i % p == 0:
            for m in range(k):
                for n in range(k):
                    if m != n:
                        w_ml = mlp_models[m * p + variable_index].weight
                        w_nl = mlp_models[n * p + variable_index].weight
                        penalty2 += (1 - torch.exp(-w_ml**2 / 0.1)) * (1 - torch.exp(-w_nl**2 / 0.1))

        total_loss = loss + 0.01 * penalty1 + 0.1 * penalty2 + 0.001 * penalty3
        total_loss.backward()
        optimizer.step()

    # Update labels using k-means
    class_weights = [np.zeros((p, p-1)) for _ in range(k)]
    for i, model in enumerate(mlp_models):
        class_label = i // p
        variable_index = i % p
        weights = model.weight.detach().numpy()
        class_weights[class_label][variable_index, :] = weights

    extended_class_weights = expand_weight_matrices(class_weights)
    labels = cluster.kmeans(X, k, extended_class_weights)  # Update labels based on current weights

    print(f"Epoch {epoch+1}, Loss: {total_loss.item()}")

# Now the model is trained with dynamic updates to the class labels.

# Collect final weights and reshape them
networks = []


class_weights = [np.zeros((p, p-1)) for _ in range(k)]
for i, model in enumerate(mlp_models):
    class_label = i // p
    variable_index = i % p
    weights = model.weight.detach().cpu().numpy()
    class_weights[class_label][variable_index, :] = weights

extended_class_weights = expand_weight_matrices(class_weights)
networks.extend(extended_class_weights)

# Store the final labels
subgroups = labels.tolist()  # Convert numpy array to list if needed for compatibility.


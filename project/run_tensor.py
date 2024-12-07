"""
Be sure you have minitorch installed in you Virtual Env.
>>> pip install -Ue .
"""

import minitorch

# Use this function to make a random parameter in
# your module.
def RParam(*shape):
    r = 2 * (minitorch.rand(shape) - 0.5)
    return minitorch.Parameter(r)

# TODO: Implement for Task 2.5.
class Linear(minitorch.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = RParam(in_features, out_features)
        self.bias = RParam(out_features)

    def __call__(self, x):
        x_expanded = x.view(x.shape[0], x.shape[1], 1)  # Shape: (x.shape[0], x.shape[0], 1)

        weights_expanded = self.weights.value.view(1, self.weights.value.shape[0], self.weights.value.shape[1])

        # Element-wise multiplication and sum over 'x.shape[1]' axis (x.shape[1] == self.weights.value.shape[0])
        output = (x_expanded * weights_expanded).sum(dim=1)  # Shape: (x.shape[1], self.weights.value.shape[1])

        # Add bias (broadcasted over batch_size)
        output = output + self.bias.value  # Shapes: (batch_size, out_features) + (out_features,)

        return output.view(x.shape[0], self.weights.value.shape[1])

class Network(minitorch.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # Initialize weights and biases
        self.hidden_size = hidden_size
        self.layer1 = Linear(2, hidden_size)
        self.layer2 = Linear(hidden_size, hidden_size)
        self.layer3 = Linear(hidden_size, 1)

    def forward(self, x):
        # Define the forward pass
        x = self.layer1(x).relu()
        x = self.layer2(x).relu()
        x = self.layer3(x).sigmoid()
        return x

def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)


class TensorTrain:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers
        self.model = Network(hidden_layers)

    def run_one(self, x):
        return self.model.forward(minitorch.tensor([x]))

    def run_many(self, X):
        return self.model.forward(minitorch.tensor(X))

    def train(self, data, learning_rate, max_epochs=500, log_fn=default_log_fn):
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.model = Network(self.hidden_layers)
        optim = minitorch.SGD(self.model.parameters(), learning_rate)

        X = minitorch.tensor(data.X)
        y = minitorch.tensor(data.y)

        losses = []
        for epoch in range(1, self.max_epochs + 1):
            total_loss = 0.0
            correct = 0
            optim.zero_grad()

            # Forward
            out = self.model.forward(X).view(data.N)
            prob = (out * y) + (out - 1.0) * (y - 1.0)

            loss = -prob.log()
            (loss / data.N).sum().view(1).backward()
            total_loss = loss.sum().view(1)[0]
            losses.append(total_loss)

            # Update
            optim.step()

            # Logging
            if epoch % 10 == 0 or epoch == max_epochs:
                y2 = minitorch.tensor(data.y)
                correct = int(((out.detach() > 0.5) == y2).sum()[0])
                log_fn(epoch, total_loss, correct, losses)


if __name__ == "__main__":
    PTS = 50
    HIDDEN = 20
    RATE = 0.5
    data = minitorch.datasets["Simple"](PTS)
    TensorTrain(HIDDEN).train(data, RATE)

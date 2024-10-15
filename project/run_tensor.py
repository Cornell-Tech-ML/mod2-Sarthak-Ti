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

def default_log_fn(epoch, total_loss, correct, losses):
    print("Epoch ", epoch, " loss ", total_loss, "correct", correct)

class Network(minitorch.Module):
    def __init__(self, hidden_layers):
        super().__init__()
        #we call 3 linear layers
        self.layer1 = Linear(2, hidden_layers)
        self.layer2 = Linear(hidden_layers, hidden_layers)
        self.layer3 = Linear(hidden_layers, 1)
    
    def forward(self, x):
        x = self.layer1(x).relu()
        # print(x.shape)
        x = self.layer2(x).relu()
        # print(x.shape)
        x = self.layer3(x)
        # print(x.shape)
        return x.sigmoid()
    
class Linear(minitorch.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        #now we use tensors to store the weights and biases
        self.weights = RParam(in_size, out_size)
        self.bias = RParam(out_size)
    
    def forward(self, inputs):
        #we use the matrix multiplication
        # return minitorch.dot(inputs, self.weights) + self.bias
        #but we don't have matrix multiplication so we can't actually do this! 
        #let's say we have x a 1x30 tensor, and have a weight tensor of 30x100
        #the fast way is to just matmul of x*weight, but we can't do that
        #this can be done the same with broadcasting and summation!
        # print(inputs.shape) #the input tensor is 50x2 or whatever the input dimension is
        #but we have a bathc size of 50 in this case
        # print(inputs.shape, self.weights.value.shape, self.bias.value.shape) #50x2, 2x2, 2
        #one option is a for loop for every elementin the batch where we just do it, but that's not efficient
        #let's use broadcasting, if we make it 50x1x2, that's duplicated so it's 50x2x2, and that's what we want!
        # print(self.weights.value.shape)
        expanded = inputs.view([inputs.shape[0], 1, inputs.shape[1]])
        mul = expanded * self.weights.value.permute(1, 0)
        # print(mul.shape)
        # print(mul.shape,expanded.shape) #is 50x2x2 after the multiplication, now we sum along axis 1
        # print(mul.shape)
        # print(mul.sum(1).shape) #50x2
        # print(inputs.shape[0], self.weights.value.shape[1])
        sums = mul.sum(2).view([inputs.shape[0], self.weights.value.shape[1]])
        # print(sums.shape) #again 50 x 2
        #and now we add the bias
        biased = sums + self.bias.value
        # print('finished')
        return biased
        


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
    HIDDEN = 3
    RATE = 1
    data = minitorch.datasets["Diag"](PTS)
    TensorTrain(HIDDEN).train(data, RATE, max_epochs=2000)

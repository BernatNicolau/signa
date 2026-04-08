import torch  ## torch let's us create tensors and also provides helper functions
import torch.nn as nn  ## torch.nn gives us nn.Module(), nn.Embedding() and nn.Linear()
import torch.optim as optim


class myNN_auto(nn.Module):
    def __init__(self):
        super().__init__()

        self.stack = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, input_values):

        return self.stack(input_values)


model = myNN_auto()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

x = torch.tensor([0.0, 0.5, 1.0]).view(-1, 1)
y = torch.tensor([0.0, 1.0, 0.0]).view(-1, 1)


for epoch in range(2000):
    output = model(x)
    loss = criterion(output, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

print(model(x).detach().numpy())


import matplotlib.pyplot as plt

# Create 100 points from 0 to 1 to see the smooth curve
test_x = torch.linspace(-1, 2, 100).view(-1, 1)

# 1. Get the output of the FIRST layer + ReLU only
with torch.no_grad():
    # We stop the model halfway through
    hidden_layer_output = model.stack[0](test_x)
    activated_output = model.stack[1](hidden_layer_output)  # The ReLUs

# 2. Plot the 16 individual "hinges"
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(test_x.numpy(), activated_output.numpy())
plt.title("The 16 Individual ReLU 'Hinges'")
plt.xlabel("Input x")

# 3. Plot the final result (The sum)
plt.subplot(1, 2, 2)
final_output = model(test_x)
plt.plot(test_x.numpy(), final_output.detach().numpy(), color="red", linewidth=3)
plt.scatter(x, y, color="black")  # Plot your 3 original dots
plt.title("The Final Combined 'Mountain'")
plt.xlabel("Input x")

plt.tight_layout()
plt.show()


weights = model.stack[2].weight.detach().numpy()
print("Importance of each of the 16 neurons:")
print(weights)

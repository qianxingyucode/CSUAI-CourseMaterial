class EMNISTNet(nn.Module):
    def __init__(self, print_toggle):
        super().__init__()
        self.print_toggle = print_toggle

        # Conv1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(num_features=64)

        # Conv2
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.bnorm2 = nn.BatchNorm2d(num_features=128) # Input: number of channels

        # Conv3
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3)
        self.bnorm3 = nn.BatchNorm2d(num_features=256) # Input: number of channels

        self.fc1 = nn.Linear(in_features=2*2*256, out_features=256)
        self.fc2 = nn.Linear(in_features=256, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=26)

    def forward(self, x):
        if self.print_toggle:
            print(f"Input: {list(x.shape)}")

        # First Block: conv -> max_pool -> bnorm -> relu
        x = F.max_pool2d(self.conv1(x), 2)
        x = F.leaky_relu((self.bnorm1(x)))
        x = F.dropout(x, p=0.25, training=self.training)
        if self.print_toggle:
            print(f"First Block: {list(x.shape)}")

        # Second Block: conv -> max_pool -> bnorm -> relu
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.leaky_relu((self.bnorm2(x)))
        x = F.dropout(x, p=0.25, training=self.training)
        if self.print_toggle:
            print(f"Second Block: {list(x.shape)}")

        # Third Block: conv -> max_pool -> bnorm -> relu
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.leaky_relu((self.bnorm3(x)))
        x = F.dropout(x, p=0.25, training=self.training)
        if self.print_toggle:
            print(f"Second Block: {list(x.shape)}")

        # Reshape for linear layer
        n_units = x.shape.numel() / x.shape[0]
        x = x.view(-1, int(n_units))
        if self.print_toggle:
            print(f"Vectorized: {list(x.shape)}")

        # Linear layers
        x = F.leaky_relu(self.fc1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.leaky_relu(self.fc2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.fc3(x)
        if self.print_toggle:
            print(f"Final Output: {list(x.shape)}")

        return x

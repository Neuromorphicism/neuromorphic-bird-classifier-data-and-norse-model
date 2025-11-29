# Most of this code comes from: https://norse.github.io/notebooks/poker-dvs_classifier.html
# Comments in [] indicate if changes were made
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import tonic
from tonic import Dataset, transforms
from aestream import FileInput
# import faery
from tqdm import tqdm, trange
import torchvision
from norse.torch import LIFParameters, LIFState
from norse.torch.module.lif import LIFCell, LIFRecurrentCell

# Original comment: Notice the difference between "LIF" (leaky integrate-and-fire) and "LI" (leaky integrator) [NOT CHANGED]
from norse.torch import LICell, LIState
from typing import NamedTuple

# Neuromorphicism comment: for our simulated events from webm this might be the height of a frame [CHANGED]
sensor_size = (640, 360, 2,) 
frame_transform = tonic.transforms.ToFrame(sensor_size=sensor_size, time_window=1000)


## START OF code from Tonic website [NOT CHANGED]
class MyRecordings(Dataset):
    ordering = (
        "txyp"  # the order in which your event channels are provided in your recordings
    )
    # [CHANGED]
    sensor_size = (640, 360, 2,) 
    classes = {"chicken": 0, "eagle": 1, "owl": 2, "pigeon": 3, "raven": 4, "stork": 5}

    
    def __init__(
        self,
        train=True,
        transform=None,
        target_transform=None,
    ):
        super(MyRecordings, self).__init__(
            save_to='./data', transform=transform, target_transform=target_transform
        )
        self.train = train
        
        n_recordings = 6

        # replace the strings with your training/testing file locations or pass as an argument
        if train:
            # [CHANGED]
            # Neuromorphicism comment: Now one has to use AEStream to change .DAT files to NUMPY
            
            for key, value in self.classes.items():
                for j in range(n_recordings):
                    events = FileInput(f"../event-birds/event-{key}/{key}-{j + 1}.dat", (640, 360)).load()
                    
                    # Neuromorphicism comment: That simulated DAT file ofc is different than expected format so labels t, x, y, p have to created
                    events_np = np.array(events.tolist()) 
                    
                    events_struct = np.zeros(events_np.shape[0],
                             dtype=[("t", np.int64), ("x", np.int16),
                                    ("y", np.int16), ("p", np.int8)])
                    events_struct["t"] = events_np[:, 0]
                    events_struct["x"] = events_np[:, 1]
                    events_struct["y"] = events_np[:, 2]
                    events_struct["p"] = events_np[:, 3]
                    
                    # Now events with labels can be saved to numpy file
                    np.save(f"../numpy-data/{key}-{j + 1}.npy", events_struct)
                    print("Loaded events:", events_struct)
                

            # Neuromorphicism comment: Well the AEStream fails to build on Arm macOS so I will change this code above to Faery below (I am quite glad that you used Rust in Faery)
            
            # for j in range(n_recordings):
                # np.save(f"../numpy-data/chicken-{j + 1}.npy", faery.events_stream_from_file(faery.dirname.parent / "event-birds" / "event-chicken" / f"chicken-{j + 1}.dat").to_array())

            # Neuromorphicism comment: Oh but now Faery does not want my 2D event type from simulated event camera DAT files... what a pity! Going back to building AEStream
            # Apparently AEStream can not yet be used with Python 3.12 so thankfully Conda exists yh that is not a problem the problem is using C++ builds on macOS

            # FIX: use Linux or Windows with x64 then prepare your own numpy files from generated dat files by using AEStream not Faery. If you are on macOS comment out the FileInput AEStream for loop code above!


            self.filenames = [
                f"../numpy-data/{key}-{i + 1}.npy"
                for key, value in self.classes.items()
                for i in range(n_recordings)
            ]
        else:
            # [CHANGED!!!] Why even put such code into a documentation? If we prepare a test set that has a train False flag it just destroys the entire run...
            # raise NotImplementedError
            
            for key, value in self.classes.items():
                for j in range(n_recordings):
                    events = FileInput(f"../event-birds/event-{key}/{key}-{j + 1}.dat", (640, 360)).load()
                
                # Neuromorphicism comment: That simulated DAT file ofc is different than expected format so labels t, x, y, p have to created
                events_np = np.array(events.tolist()) 
                
                events_struct = np.zeros(events_np.shape[0],
                         dtype=[("t", np.int64), ("x", np.int16),
                                ("y", np.int16), ("p", np.int8)])
                events_struct["t"] = events_np[:, 0]
                events_struct["x"] = events_np[:, 1]
                events_struct["y"] = events_np[:, 2]
                events_struct["p"] = events_np[:, 3]
                
                # Now events with labels can be saved to numpy file
                np.save(f"../numpy-data/{key}-{j + 1}.npy", events_struct)
                print("Loaded events:", events_struct)

            
            self.filenames = [
                f"../numpy-data/{key}-{i + 1}.npy"
                for key, value in self.classes.items()
                for i in range(n_recordings)
            ]
            
    def __getitem__(self, index):
        # [CHANGED] 
        events = np.load(self.filenames[index])
        
        # [CHANGED]
        # Get the base file name without directories
        filepath = self.filenames[index]
        filename = os.path.basename(filepath)
        # Remove the file extension
        name_without_ext = os.path.splitext(filename)[0]
        # Extract the part before the first hyphen, if any
        extracted_name = name_without_ext.split('-')[0]
        
        label = self.classes[extracted_name]
        # print(label, extracted_name, filepath)

        if self.transform is not None:
            events = self.transform(events)
        
        # [CHANGED]
        return events, label

    def __len__(self):
        return len(self.filenames)

dataset = MyRecordings(train=True, transform=transforms.NumpyAsType(int))
#events = dataset[5]

dataloader = torch.utils.data.DataLoader(dataset, shuffle=True)
events = next(iter(dataloader))
## END OF code from Tonic website



## START OF Neuromorphicism code [CHANGED]
#events here have no names in our custom DAT files so this Tonic util will not work
#tonic.utils.plot_event_grid(events)

trainset = MyRecordings(train=True)
testset = MyRecordings(transform=frame_transform, train=False)
## END OF Neuromorphicism code

# Original comment: reduce this number if you run out of GPU memory [NOT CHANGED]
BATCH_SIZE = 32

# Original comment: add sparse transform to trainset, previously omitted because we wanted to look at raw events [NOT CHANGED]
trainset.transform = frame_transform

train_loader = torch.utils.data.DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    collate_fn=tonic.collation.PadTensors(batch_first=False),
    shuffle=False,
)


# DEFINE SPIKING NEURAL NETWORK

class SNNState(NamedTuple):
    lif0: LIFState
    readout: LIState


class SNN(torch.nn.Module):
    def __init__(
        self,
        input_features,
        hidden_features,
        output_features,
        tau_syn_inv,
        tau_mem_inv,
        record=False,
        dt=1e-3,
    ):
        super(SNN, self).__init__()
        self.l1 = LIFRecurrentCell(
            input_features,
            hidden_features,
            p=LIFParameters(
                alpha=100,
                v_th=torch.as_tensor(0.3),
                tau_syn_inv=tau_syn_inv,
                tau_mem_inv=tau_mem_inv,
            ),
            dt=dt,
        )
        self.input_features = input_features
        self.fc_out = torch.nn.Linear(hidden_features, output_features, bias=False)
        self.out = LICell(dt=dt)

        self.hidden_features = hidden_features
        self.output_features = output_features
        self.record = record

    def forward(self, x):
        seq_length, batch_size, _, _, _ = x.shape
        s1 = so = None
        voltages = []

        if self.record:
            self.recording = SNNState(
                LIFState(
                    z=torch.zeros(seq_length, batch_size, self.hidden_features),
                    v=torch.zeros(seq_length, batch_size, self.hidden_features),
                    i=torch.zeros(seq_length, batch_size, self.hidden_features),
                ),
                LIState(
                    v=torch.zeros(seq_length, batch_size, self.output_features),
                    i=torch.zeros(seq_length, batch_size, self.output_features),
                ),
            )

        for ts in range(seq_length):
            z = x[ts, :, :, :].view(-1, self.input_features)
            z, s1 = self.l1(z, s1)
            z = self.fc_out(z)
            vo, so = self.out(z, so)
            if self.record:
                self.recording.lif0.z[ts, :] = s1.z
                self.recording.lif0.v[ts, :] = s1.v
                self.recording.lif0.i[ts, :] = s1.i
                self.recording.readout.v[ts, :] = so.v
                self.recording.readout.i[ts, :] = so.i
            voltages += [vo]

        return torch.stack(voltages)


example_snn = SNN(
    np.product(trainset.sensor_size),
    100,
    len(trainset.classes),
    tau_syn_inv=torch.tensor(1 / 1e-2),
    tau_mem_inv=torch.tensor(1 / 1e-2),
    record=True,
    dt=1e-3,
)

frames, target = next(iter(train_loader))
frames[:, :1].shape

example_readout_voltages = example_snn(frames[:, :1])
voltages = example_readout_voltages.squeeze(1).detach().numpy()

plt.plot(voltages)
plt.ylabel("Voltage [a.u.]")
plt.xlabel("Time [us]")
plt.show()

plt.plot(example_snn.recording.lif0.v.squeeze(1).detach().numpy())
plt.show()

plt.plot(example_snn.recording.lif0.i.squeeze(1).detach().numpy())
plt.show()


# TRAINING SNN
def decode(x):
    x, _ = torch.max(x, 0)
    log_p_y = torch.nn.functional.log_softmax(x, dim=1)
    return log_p_y


class Model(torch.nn.Module):
    def __init__(self, snn, decoder):
        super(Model, self).__init__()
        self.snn = snn
        self.decoder = decoder

    def forward(self, x):
        x = self.snn(x)
        log_p_y = self.decoder(x)
        return log_p_y

LR = 0.002
INPUT_FEATURES = np.product(trainset.sensor_size)
HIDDEN_FEATURES = 100
OUTPUT_FEATURES = len(trainset.classes)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

model = Model(
    snn=SNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
        tau_syn_inv=torch.tensor(1 / 1e-2),
        tau_mem_inv=torch.tensor(1 / 1e-2),
    ),
    decoder=decode,
).to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
model

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    losses = []

    for (data, target) in tqdm(train_loader, leave=False):
        data, target = data.to(device), torch.LongTensor(target).to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = torch.nn.functional.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    mean_loss = np.mean(losses)
    return losses, mean_loss

# Neuromorphicism comment: Now imagine writing all this yourself each time from scratch

def test(model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), torch.LongTensor(target).to(device)
            output = model(data)
            test_loss += torch.nn.functional.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)

    return test_loss, accuracy

training_losses = []
mean_losses = []
test_losses = []
accuracies = []

torch.autograd.set_detect_anomaly(True)

EPOCHS = 10

for epoch in trange(EPOCHS):
    training_loss, mean_loss = train(model, DEVICE, train_loader, optimizer, epoch)
    test_loss, accuracy = test(model, DEVICE, test_loader, epoch)
    training_losses += training_loss
    mean_losses.append(mean_loss)
    test_losses.append(test_loss)
    accuracies.append(accuracy)

print(f"final accuracy: {accuracies[-1]}")

# Neuromorphicism comment: Finally after 250 lines of Python there is a trained model but what about inference?
# Neuromorphicism comment: This line below breaks a program if you do not have CUDA, stop implying that everyone has it... [CHANGED!]
#trained_readout_voltages = trained_snn(frames[:, :1].to("cuda"))

# Neuromorphicism comment: Detect device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Neuromorphicism comment: Move input and model to device
torch.save(model.state_dict(), "../saved-snn-models/model.pth")

trained_snn = model.snn.to(device)
input_tensor = frames[:, :1].to(device)
trained_readout_voltages = trained_snn(input_tensor)

# [NOT CHANGED]
plt.plot(trained_readout_voltages.squeeze(1).cpu().detach().numpy())

plt.ylabel("Voltage [a.u.]")
plt.xlabel("Time [ms]")
plt.show()
